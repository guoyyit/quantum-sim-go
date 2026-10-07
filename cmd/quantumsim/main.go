// Command quantumsim simulates the 2D time-dependent Schroedinger equation.
//
//	quantumsim run   -config run.json     real-time evolution (resumable)
//	quantumsim ground -config ground.json  ground state via imaginary time
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"

	"github.com/guoyyit/quantum-sim-go/internal/sim"
)

// fileConfig mirrors sim.Config plus physical setup, as JSON.
type fileConfig struct {
	Nx              int     `json:"nx"`
	Ny              int     `json:"ny"`
	Lx              float64 `json:"lx"`
	Ly              float64 `json:"ly"`
	Mass            float64 `json:"mass"`
	Hbar            float64 `json:"hbar"`
	Dt              float64 `json:"dt"`
	Steps           int     `json:"steps"`
	Workers         int     `json:"workers"`
	OutputDir       string  `json:"outputDir"`
	SnapshotEvery   int     `json:"snapshotEvery"`
	CheckpointEvery int     `json:"checkpointEvery"`
	NormDriftTol    float64 `json:"normDriftTol"`
	EnergyDriftTol  float64 `json:"energyDriftTol"`

	Potential struct {
		Type    string  `json:"type"` // "zero" | "harmonic" | "barrier"
		Omega   float64 `json:"omega"`
		CenterX float64 `json:"centerX"`
		CenterY float64 `json:"centerY"`
		Height  float64 `json:"height"`
		X0      float64 `json:"x0"`
		X1      float64 `json:"x1"`
	} `json:"potential"`
	Initial struct {
		Type  string  `json:"type"` // "gaussian" | "plane" | "superposition"
		X0    float64 `json:"x0"`
		Y0    float64 `json:"y0"`
		Sigma float64 `json:"sigma"`
		Kx    float64 `json:"kx"`
		Ky    float64 `json:"ky"`
		K2x   float64 `json:"k2x"`
		K2y   float64 `json:"k2y"`
	} `json:"initial"`
	Ground struct {
		Tol      float64 `json:"tol"`
		MaxSteps int     `json:"maxSteps"`
	} `json:"ground"`
}

func (fc *fileConfig) config() sim.Config {
	return sim.Config{
		Nx: fc.Nx, Ny: fc.Ny, Lx: fc.Lx, Ly: fc.Ly,
		Mass: fc.Mass, Hbar: fc.Hbar, Dt: fc.Dt, Steps: fc.Steps,
		Workers: fc.Workers, OutputDir: fc.OutputDir,
		SnapshotEvery: fc.SnapshotEvery, CheckpointEvery: fc.CheckpointEvery,
		NormDriftTol: fc.NormDriftTol, EnergyDriftTol: fc.EnergyDriftTol,
	}
}

func (fc *fileConfig) potential(g *sim.Grid) ([]float64, error) {
	switch fc.Potential.Type {
	case "zero", "":
		return sim.ZeroPotential(g), nil
	case "harmonic":
		return sim.HarmonicPotential(g, fc.Mass, fc.Potential.Omega, fc.Potential.CenterX, fc.Potential.CenterY), nil
	case "barrier":
		return sim.BarrierPotential(g, fc.Potential.Height, fc.Potential.X0, fc.Potential.X1), nil
	}
	return nil, fmt.Errorf("unknown potential type %q", fc.Potential.Type)
}

func (fc *fileConfig) initial(g *sim.Grid) (*sim.Field, error) {
	switch fc.Initial.Type {
	case "gaussian", "":
		return sim.GaussianPacket(g, fc.Initial.X0, fc.Initial.Y0, fc.Initial.Sigma, fc.Initial.Kx, fc.Initial.Ky), nil
	case "plane":
		return sim.PlaneWave(g, fc.Initial.Kx, fc.Initial.Ky), nil
	case "superposition":
		f := sim.AddFields(
			sim.PlaneWave(g, fc.Initial.Kx, fc.Initial.Ky),
			sim.PlaneWave(g, fc.Initial.K2x, fc.Initial.K2y))
		sim.Normalize(f, g)
		return f, nil
	}
	return nil, fmt.Errorf("unknown initial type %q", fc.Initial.Type)
}

func loadConfig(path string) (*fileConfig, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var fc fileConfig
	if err := json.Unmarshal(b, &fc); err != nil {
		return nil, fmt.Errorf("parsing %s: %w", path, err)
	}
	return &fc, nil
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: quantumsim <run|ground> -config config.json")
		os.Exit(2)
	}
	fs := flag.NewFlagSet(os.Args[1], flag.ExitOnError)
	cfgPath := fs.String("config", "", "JSON configuration file")
	fs.Parse(os.Args[2:])
	if *cfgPath == "" {
		fmt.Fprintln(os.Stderr, "missing -config")
		os.Exit(2)
	}
	fc, err := loadConfig(*cfgPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(2)
	}
	cfg := fc.config()
	if err := cfg.Validate(); err != nil {
		fmt.Fprintln(os.Stderr, "invalid config:", err)
		os.Exit(2)
	}
	g := sim.NewGrid(cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly)
	V, err := fc.potential(g)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(2)
	}
	psi0, err := fc.initial(g)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(2)
	}

	switch os.Args[1] {
	case "run":
		os.Exit(cmdRun(cfg, V, psi0))
	case "ground":
		os.Exit(cmdGround(cfg, fc, V, psi0))
	default:
		fmt.Fprintln(os.Stderr, "unknown command", os.Args[1])
		os.Exit(2)
	}
}

func cmdRun(cfg sim.Config, V []float64, psi0 *sim.Field) int {
	res, err := sim.Run(cfg, V, psi0)
	for _, d := range res.History {
		fmt.Printf("step=%d t=%.4g norm=%.12f energy=%.12f status=%s\n",
			d.Step, d.Time, d.Norm, d.Energy, d.Status)
	}
	var de *sim.DriftError
	if errors.As(err, &de) {
		fmt.Fprintln(os.Stderr, "conservation drift:", de)
		return 3
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "run failed:", err)
		return 1
	}
	fmt.Println("status:", res.Status)
	return 0
}

func cmdGround(cfg sim.Config, fc *fileConfig, V []float64, psi0 *sim.Field) int {
	tol := fc.Ground.Tol
	if tol <= 0 {
		tol = 1e-10
	}
	maxSteps := fc.Ground.MaxSteps
	if maxSteps <= 0 {
		maxSteps = 100000
	}
	res, err := sim.GroundState(cfg, V, psi0, tol, maxSteps)
	if err != nil {
		fmt.Fprintln(os.Stderr, "ground state failed:", err)
		return 1
	}
	fmt.Printf("energy=%.12f steps=%d converged=%v finalResidual=%.3g\n",
		res.Energy, res.Steps, res.Converged, res.Residuals[len(res.Residuals)-1])
	if cfg.OutputDir != "" {
		path, err := sim.WriteSnapshot(cfg.OutputDir, &sim.Snapshot{
			Step: 0, Nx: res.Psi.Nx, Ny: res.Psi.Ny, Data: res.Psi.Data,
		})
		if err != nil {
			fmt.Fprintln(os.Stderr, "writing ground state:", err)
			return 1
		}
		fmt.Println("ground state written to", path)
	}
	if !res.Converged {
		return 4
	}
	return 0
}
