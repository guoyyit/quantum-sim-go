// Package solver evolves the time-dependent Schroedinger equation
//
//	i hbar d/dt psi = [ -hbar^2/(2m) laplacian + V ] psi
//
// with a second-order Strang split-operator spectral method. The
// propagator is exactly unitary, so the norm is conserved to roundoff and
// the energy error stays bounded (no secular growth) for long runs.
package solver

import (
	"fmt"
	"math"
	"math/cmplx"

	"github.com/guoyyit/quantum-sim-go/internal/fft"
	"github.com/guoyyit/quantum-sim-go/internal/grid"
	"github.com/guoyyit/quantum-sim-go/internal/observables"
	"github.com/guoyyit/quantum-sim-go/internal/potential"
	"github.com/guoyyit/quantum-sim-go/internal/wavefunc"
)

// Status classifies the trustworthiness of a run.
type Status int

const (
	// StatusOK means all monitored conserved quantities stayed within
	// their tolerances.
	StatusOK Status = iota
	// StatusNormDrift means the norm drift exceeded Params.NormTol.
	StatusNormDrift
	// StatusEnergyDrift means the energy drift exceeded Params.EnergyTol.
	StatusEnergyDrift
)

// String renders the status for diagnostics.
func (s Status) String() string {
	switch s {
	case StatusOK:
		return "ok"
	case StatusNormDrift:
		return "norm-drift"
	case StatusEnergyDrift:
		return "energy-drift"
	}
	return fmt.Sprintf("status(%d)", int(s))
}

// Params controls time evolution.
type Params struct {
	Dt          float64 // time step
	Steps       int     // number of steps to evolve
	Mass        float64 // particle mass (default 1)
	HBar        float64 // reduced Planck constant (default 1)
	SampleEvery int     // sampling period in steps (default 1)
	Workers     int     // parallel workers (default 1); output is identical for any value
	NormTol     float64 // max allowed |norm-1| before StatusNormDrift (default 1e-6)
	EnergyTol   float64 // max allowed relative energy drift before StatusEnergyDrift (default 1e-3)
}

// Frame is a sampled wavefunction at a given step.
type Frame struct {
	Step int
	Time float64
	Psi  *wavefunc.WaveFunction
}

// Hooks are callbacks invoked during a run. Sample receives a copy of the
// wavefunction at sampling steps; AfterStep receives the live wavefunction
// after every step and must not retain or mutate it.
type Hooks struct {
	Sample    func(f Frame)
	AfterStep func(step int, time float64, psi *wavefunc.WaveFunction)
}

// Solver performs split-operator time evolution.
type Solver struct {
	g     *grid.Grid
	pot   *potential.Potential
	p     Params
	halfV []complex128 // exp(-i V dt / (2 hbar))
	kin   []complex128 // exp(-i hbar k^2 dt / (2 m))

	// MaxNormDrift and MaxEnergyDrift record the largest drifts observed
	// during the last Run, for diagnostics.
	MaxNormDrift   float64
	MaxEnergyDrift float64
}

// New builds a solver and precomputes evolution phases. Grid dimensions
// must be powers of two (radix-2 FFT).
func New(g *grid.Grid, pot *potential.Potential, p Params) *Solver {
	if g.Nx&(g.Nx-1) != 0 || g.Ny&(g.Ny-1) != 0 {
		panic("solver: grid dimensions must be powers of two")
	}
	if p.Dt <= 0 {
		panic("solver: Dt must be positive")
	}
	if p.Mass == 0 {
		p.Mass = 1
	}
	if p.HBar == 0 {
		p.HBar = 1
	}
	if p.SampleEvery <= 0 {
		p.SampleEvery = 1
	}
	if p.Workers <= 0 {
		p.Workers = 1
	}
	if p.NormTol == 0 {
		p.NormTol = 1e-6
	}
	if p.EnergyTol == 0 {
		p.EnergyTol = 1e-3
	}
	s := &Solver{g: g, pot: pot, p: p}
	n := g.Nx * g.Ny
	s.halfV = make([]complex128, n)
	for i, v := range pot.V {
		s.halfV[i] = cmplx.Exp(complex(0, -v*p.Dt/(2*p.HBar)))
	}
	s.kin = make([]complex128, n)
	for i := 0; i < g.Nx; i++ {
		kx := g.Kx[i]
		for j := 0; j < g.Ny; j++ {
			ky := g.Ky[j]
			e := p.HBar * (kx*kx + ky*ky) * p.Dt / (2 * p.Mass)
			s.kin[i*g.Ny+j] = cmplx.Exp(complex(0, -e))
		}
	}
	return s
}

// Step advances psi by one time step in place using the Strang sequence
// half-potential, full kinetic, half-potential.
func (s *Solver) Step(psi *wavefunc.WaveFunction) {
	d := psi.Data
	for i := range d {
		d[i] *= s.halfV[i]
	}
	fft.FFT2D(d, s.g.Nx, s.g.Ny, false, s.p.Workers)
	for i := range d {
		d[i] *= s.kin[i]
	}
	fft.FFT2D(d, s.g.Nx, s.g.Ny, true, s.p.Workers)
	for i := range d {
		d[i] *= s.halfV[i]
	}
}

// Run evolves psi from startStep for p.Steps steps. It samples frames and
// monitors norm and energy conservation, returning a non-OK status if a
// drift tolerance is violated. The wavefunction is left at the final step.
func (s *Solver) Run(psi *wavefunc.WaveFunction, startStep int, h Hooks) Status {
	norm0 := psi.Norm2(s.g.Dx, s.g.Dy)
	e0 := observables.Energy(psi, s.g, s.pot, s.p.Mass, s.p.HBar)
	status := StatusOK
	s.MaxNormDrift, s.MaxEnergyDrift = 0, 0

	if h.Sample != nil && startStep%s.p.SampleEvery == 0 {
		h.Sample(Frame{Step: startStep, Time: float64(startStep) * s.p.Dt, Psi: psi.Clone()})
	}
	for step := startStep + 1; step <= startStep+s.p.Steps; step++ {
		s.Step(psi)
		t := float64(step) * s.p.Dt
		if h.AfterStep != nil {
			h.AfterStep(step, t, psi)
		}
		if h.Sample != nil && step%s.p.SampleEvery == 0 {
			h.Sample(Frame{Step: step, Time: t, Psi: psi.Clone()})
		}
		// Conservation monitoring at sampling points.
		if step%s.p.SampleEvery == 0 || step == startStep+s.p.Steps {
			nd := math.Abs(math.Sqrt(psi.Norm2(s.g.Dx, s.g.Dy)/norm0) - 1)
			e := observables.Energy(psi, s.g, s.pot, s.p.Mass, s.p.HBar)
			ed := math.Abs(e-e0) / math.Max(math.Abs(e0), 1e-300)
			if nd > s.MaxNormDrift {
				s.MaxNormDrift = nd
			}
			if ed > s.MaxEnergyDrift {
				s.MaxEnergyDrift = ed
			}
			if status == StatusOK && nd > s.p.NormTol {
				status = StatusNormDrift
			}
			if status == StatusOK && ed > s.p.EnergyTol {
				status = StatusEnergyDrift
			}
		}
	}
	return status
}
