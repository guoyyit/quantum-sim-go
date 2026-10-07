package sim

import (
	"fmt"
	"math"
)

// GroundStateResult reports the outcome of imaginary-time evolution.
type GroundStateResult struct {
	Energy    float64
	Psi       *Field
	Residuals []float64 // |E_n - E_{n-1}| per step (the residual curve)
	Steps     int
	Converged bool
}

// GroundState evolves psi0 in imaginary time (tau = i*t) until the energy
// residual drops below tol or maxSteps is reached. Imaginary-time Strang
// splitting damps excited states exponentially, so the normalized state
// converges to the ground state and <H> to the ground-state energy.
// cfg.Dt is used as the imaginary time step.
func GroundState(cfg Config, V []float64, psi0 *Field, tol float64, maxSteps int) (*GroundStateResult, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if len(V) != cfg.Nx*cfg.Ny {
		return nil, fmt.Errorf("sim: potential has %d points, grid needs %d", len(V), cfg.Nx*cfg.Ny)
	}
	if psi0 == nil || psi0.Nx != cfg.Nx || psi0.Ny != cfg.Ny {
		return nil, fmt.Errorf("sim: initial state does not match grid %dx%d", cfg.Nx, cfg.Ny)
	}
	solver, err := NewSolver(cfg, V) // used for observables only
	if err != nil {
		return nil, err
	}
	g := solver.Grid()

	// Imaginary-time propagators: exp(-V dtau/2h) and exp(-h k^2 dtau/2m).
	expHalfV := make([]complex128, len(V))
	for i, v := range V {
		expHalfV[i] = complex(math.Exp(-v*cfg.Dt/(2*cfg.Hbar)), 0)
	}
	expKin := make([]complex128, len(V))
	for i, k2 := range solver.Ksq() {
		expKin[i] = complex(math.Exp(-cfg.Hbar*k2*cfg.Dt/(2*cfg.Mass)), 0)
	}

	psi := psi0.Clone()
	normalize(psi, g)

	res := &GroundStateResult{Psi: psi}
	prevE := solver.EnergyExpectation(psi)
	for step := 1; step <= maxSteps; step++ {
		applyMul(psi, expHalfV, cfg.Workers)
		FFT2D(psi, false, cfg.Workers)
		applyMul(psi, expKin, cfg.Workers)
		FFT2D(psi, true, cfg.Workers)
		applyMul(psi, expHalfV, cfg.Workers)
		normalize(psi, g)

		e := solver.EnergyExpectation(psi)
		residual := math.Abs(e - prevE)
		res.Residuals = append(res.Residuals, residual)
		res.Steps = step
		prevE = e
		if residual < tol {
			res.Converged = true
			break
		}
	}
	res.Energy = prevE
	return res, nil
}

// normalize scales psi to unit L2 norm.
func normalize(psi *Field, g *Grid) {
	n := Norm(psi, g.CellArea())
	if n == 0 {
		return
	}
	inv := complex(1/math.Sqrt(n), 0)
	for i := range psi.Data {
		psi.Data[i] *= inv
	}
}

// EstimateOrder infers the convergence order p from three energies computed
// with step sizes dt, dt/2, dt/4, assuming the error scales as C*dt^p:
//
//	p = log2( |e1-e2| / |e2-e3| )
func EstimateOrder(e1, e2, e3 float64) float64 {
	d1 := math.Abs(e1 - e2)
	d2 := math.Abs(e2 - e3)
	if d2 == 0 {
		return math.Inf(1)
	}
	return math.Log2(d1 / d2)
}
