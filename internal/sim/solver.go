package sim

import (
	"fmt"
	"math"
)

// Solver evolves a Field with the Strang split-operator method:
//
//	psi(t+dt) = exp(-iV dt/2h) F^-1 exp(-i h k^2 dt/2m) F exp(-iV dt/2h) psi(t)
//
// The splitting is second-order accurate in dt and spectral in space.
// Every elementary update is unitary, so the norm is preserved exactly
// (up to rounding) regardless of step count.
type Solver struct {
	cfg      Config
	grid     *Grid
	V        []float64
	expHalfV []complex128 // exp(-i V dt / (2 hbar))
	expKin   []complex128 // exp(-i hbar k^2 dt / (2 mass))
	ksq      []float64    // kx^2 + ky^2 per grid point
}

// NewSolver builds the evolution operators for cfg and potential V
// (len(V) must be Nx*Ny, row-major like Field.Data).
func NewSolver(cfg Config, V []float64) (*Solver, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	g := NewGrid(cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly)
	if len(V) != g.Nx*g.Ny {
		return nil, fmt.Errorf("sim: potential has %d points, grid needs %d", len(V), g.Nx*g.Ny)
	}
	s := &Solver{cfg: cfg, grid: g, V: V}
	s.expHalfV = make([]complex128, len(V))
	for i, v := range V {
		s.expHalfV[i] = cmplxExp(-v * cfg.Dt / (2 * cfg.Hbar))
	}
	s.expKin = make([]complex128, len(V))
	s.ksq = make([]float64, len(V))
	for iy := 0; iy < g.Ny; iy++ {
		for ix := 0; ix < g.Nx; ix++ {
			k2 := g.Kx[ix]*g.Kx[ix] + g.Ky[iy]*g.Ky[iy]
			i := iy*g.Nx + ix
			s.ksq[i] = k2
			s.expKin[i] = cmplxExp(-cfg.Hbar * k2 * cfg.Dt / (2 * cfg.Mass))
		}
	}
	return s, nil
}

// cmplxExp returns exp(i*theta) computed with a single Sincos call so the
// result is reproducible across platforms and goroutine schedules.
func cmplxExp(theta float64) complex128 {
	s, c := math.Sincos(theta)
	return complex(c, s)
}

// Config returns the configuration the solver was built with.
func (s *Solver) Config() Config { return s.cfg }

// Grid returns the spatial grid.
func (s *Solver) Grid() *Grid { return s.grid }

// Potential returns the potential samples.
func (s *Solver) Potential() []float64 { return s.V }

// Ksq returns kx^2+ky^2 per grid point (row-major).
func (s *Solver) Ksq() []float64 { return s.ksq }

// applyMul multiplies psi elementwise by op, split across workers.
func applyMul(psi *Field, op []complex128, workers int) {
	parallelFor(psi.Ny, workers, func(iy int) {
		row := psi.Data[iy*psi.Nx : (iy+1)*psi.Nx]
		o := op[iy*psi.Nx : (iy+1)*psi.Nx]
		for i := range row {
			row[i] *= o[i]
		}
	})
}

// Step advances psi by one Strang time step (real time).
func (s *Solver) Step(psi *Field) {
	w := s.cfg.Workers
	applyMul(psi, s.expHalfV, w)
	FFT2D(psi, false, w)
	applyMul(psi, s.expKin, w)
	FFT2D(psi, true, w)
	applyMul(psi, s.expHalfV, w)
}
