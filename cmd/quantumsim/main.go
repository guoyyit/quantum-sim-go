// Command quantumsim runs 2D Schroedinger simulations.
//
// Subcommands:
//
//	quantumsim evolve -config run.json [-resume]
//	quantumsim groundstate -config run.json
//
// evolve propagates the initial wavefunction, writes sampled frames and a
// summary with conservation diagnostics, and commits checkpoints so an
// interrupted run can be resumed exactly.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"

	"github.com/guoyyit/quantum-sim-go/internal/checkpoint"
	"github.com/guoyyit/quantum-sim-go/internal/config"
	"github.com/guoyyit/quantum-sim-go/internal/grid"
	"github.com/guoyyit/quantum-sim-go/internal/groundstate"
	"github.com/guoyyit/quantum-sim-go/internal/potential"
	"github.com/guoyyit/quantum-sim-go/internal/solver"
	"github.com/guoyyit/quantum-sim-go/internal/wavefunc"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "evolve":
		cmdEvolve(os.Args[2:])
	case "groundstate":
		cmdGroundstate(os.Args[2:])
	default:
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, "usage: quantumsim <evolve|groundstate> -config run.json [-resume]\n")
}

func buildPotential(c *config.Config, g *grid.Grid) (*potential.Potential, error) {
	switch c.Potential.Kind {
	case "", "zero":
		return potential.Zero(g), nil
	case "harmonic":
		return potential.Harmonic(g, c.Potential.OmegaX, c.Potential.OmegaY), nil
	case "slit":
		return potential.SlitBarrier(g, c.Potential.BarrierHeight, c.Potential.BarrierWidth,
			c.Potential.SlitSep, c.Potential.SlitWidth), nil
	}
	return nil, fmt.Errorf("unknown potential kind %q", c.Potential.Kind)
}

func buildInitial(c *config.Config, g *grid.Grid) (*wavefunc.WaveFunction, error) {
	switch c.Initial.Kind {
	case "", "gaussian":
		sx, sy := c.Initial.SigmaX, c.Initial.SigmaY
		if sx <= 0 || sy <= 0 {
			return nil, fmt.Errorf("initial gaussian requires positive sigma_x and sigma_y")
		}
		w := wavefunc.FromFunc(g, func(x, y float64) complex128 {
			dx, dy := x-c.Initial.X0, y-c.Initial.Y0
			env := math.Exp(-dx*dx/(4*sx*sx) - dy*dy/(4*sy*sy))
			return complex(env*math.Cos(c.Initial.Kx*x+c.Initial.Ky*y),
				env*math.Sin(c.Initial.Kx*x+c.Initial.Ky*y))
		})
		w.Normalize(g.Dx, g.Dy)
		return w, nil
	}
	return nil, fmt.Errorf("unknown initial kind %q", c.Initial.Kind)
}

// writeFrame atomically writes a sampled frame as CSV: x,y,re,im.
func writeFrame(dir string, f solver.Frame, g *grid.Grid) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	name := filepath.Join(dir, fmt.Sprintf("frame-%08d.csv", f.Step))
	tmp := name + ".tmp"
	out, err := os.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	w := func(s string) { fmt.Fprintln(out, s) }
	w("x,y,re,im")
	for i := 0; i < g.Nx; i++ {
		for j := 0; j < g.Ny; j++ {
			v := f.Psi.At(i, j)
			w(fmt.Sprintf("%s,%s,%s,%s",
				strconv.FormatFloat(g.X[i], 'g', -1, 64),
				strconv.FormatFloat(g.Y[j], 'g', -1, 64),
				strconv.FormatFloat(real(v), 'g', -1, 64),
				strconv.FormatFloat(imag(v), 'g', -1, 64)))
		}
	}
	if err := out.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, name)
}

type summary struct {
	Status         string  `json:"status"`
	Steps          int     `json:"steps"`
	FinalStep      int     `json:"final_step"`
	FinalTime      float64 `json:"final_time"`
	MaxNormDrift   float64 `json:"max_norm_drift"`
	MaxEnergyDrift float64 `json:"max_energy_drift"`
}

func cmdEvolve(args []string) {
	fs := flag.NewFlagSet("evolve", flag.ExitOnError)
	cfgPath := fs.String("config", "", "path to JSON config")
	resume := fs.Bool("resume", false, "resume from the latest committed checkpoint")
	fs.Parse(args)
	if *cfgPath == "" {
		usage()
		os.Exit(2)
	}
	c, err := config.Load(*cfgPath)
	must(err)
	if c.Evolution.Dt <= 0 {
		must(fmt.Errorf("config: evolution.dt must be positive"))
	}
	g := grid.New(c.Grid.Nx, c.Grid.Ny, c.Grid.Lx, c.Grid.Ly)
	pot, err := buildPotential(c, g)
	must(err)
	psi, err := buildInitial(c, g)
	must(err)

	mgr := checkpoint.NewManager(c.Checkpoint.Dir, c.Checkpoint.Every)
	startStep := 0
	if *resume {
		st, err := mgr.LoadLatest()
		must(err)
		if st != nil {
			if st.Nx != g.Nx || st.Ny != g.Ny {
				must(fmt.Errorf("checkpoint grid %dx%d does not match config %dx%d",
					st.Nx, st.Ny, g.Nx, g.Ny))
			}
			psi = &wavefunc.WaveFunction{Data: st.Data, Nx: st.Nx, Ny: st.Ny}
			startStep = st.Step
			fmt.Printf("resumed from checkpoint at step %d (t=%g)\n", st.Step, st.Time)
		}
	}

	sv := solver.New(g, pot, solver.Params{
		Dt:          c.Evolution.Dt,
		Steps:       c.Evolution.Steps - startStep,
		SampleEvery: c.Evolution.SampleEvery,
		Workers:     c.Evolution.Workers,
		NormTol:     c.Evolution.NormTol,
		EnergyTol:   c.Evolution.EnergyTol,
	})
	hooks := solver.Hooks{
		Sample: func(f solver.Frame) {
			must(writeFrame(c.Output.Dir, f, g))
		},
		AfterStep: func(step int, t float64, p *wavefunc.WaveFunction) {
			if c.Checkpoint.Dir != "" && mgr.ShouldSave(step) {
				must(mgr.Save(checkpoint.State{
					Step: step, Time: t, Nx: g.Nx, Ny: g.Ny, Data: append([]complex128(nil), p.Data...),
				}))
			}
		},
	}
	status := sv.Run(psi, startStep, hooks)

	sum := summary{
		Status:         status.String(),
		Steps:          c.Evolution.Steps,
		FinalStep:      c.Evolution.Steps,
		FinalTime:      float64(c.Evolution.Steps) * c.Evolution.Dt,
		MaxNormDrift:   sv.MaxNormDrift,
		MaxEnergyDrift: sv.MaxEnergyDrift,
	}
	data, _ := json.MarshalIndent(sum, "", "  ")
	must(os.MkdirAll(c.Output.Dir, 0o755))
	must(os.WriteFile(filepath.Join(c.Output.Dir, "summary.json"), append(data, '\n'), 0o644))
	fmt.Println(string(data))
	if status != solver.StatusOK {
		os.Exit(1)
	}
}

func cmdGroundstate(args []string) {
	fs := flag.NewFlagSet("groundstate", flag.ExitOnError)
	cfgPath := fs.String("config", "", "path to JSON config")
	dtau := fs.Float64("dtau", 0.01, "imaginary time step")
	maxIter := fs.Int("maxiter", 20000, "maximum iterations")
	tol := fs.Float64("tol", 1e-12, "energy convergence tolerance")
	fs.Parse(args)
	if *cfgPath == "" {
		usage()
		os.Exit(2)
	}
	c, err := config.Load(*cfgPath)
	must(err)
	g := grid.New(c.Grid.Nx, c.Grid.Ny, c.Grid.Lx, c.Grid.Ly)
	pot, err := buildPotential(c, g)
	must(err)
	init, err := buildInitial(c, g)
	must(err)
	res := groundstate.Find(g, pot, groundstate.Params{
		DTau: *dtau, MaxIter: *maxIter, Tol: *tol,
	}, init)
	fmt.Printf("energy=%.12g iterations=%d converged=%v order=%.3f\n",
		res.Energy, res.Iterations, res.Converged, res.Order)
	// Persist the ground state as a frame for inspection.
	must(writeFrame(c.Output.Dir, solver.Frame{Step: 0, Time: 0, Psi: res.Psi}, g))
	// Persist the residual curve.
	out, err := os.OpenFile(filepath.Join(c.Output.Dir, "residuals.csv.tmp"),
		os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	must(err)
	fmt.Fprintln(out, "iter,residual")
	for i, r := range res.Residuals {
		fmt.Fprintf(out, "%d,%s\n", i+1, strconv.FormatFloat(r, 'g', -1, 64))
	}
	must(out.Close())
	must(os.Rename(filepath.Join(c.Output.Dir, "residuals.csv.tmp"),
		filepath.Join(c.Output.Dir, "residuals.csv")))
	if !res.Converged {
		os.Exit(1)
	}
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
