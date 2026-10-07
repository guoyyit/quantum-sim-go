// Package groundstate finds ground states via imaginary-time evolution.
//
// Replacing t by -i*tau turns the Schroedinger equation into a diffusion
// equation whose solution, after renormalization, converges to the ground
// state: psi(tau) ~ sum_n c_n exp(-E_n tau) phi_n. The same Strang
// split-operator scheme as the real-time solver is used. The converged
// state carries the O(dtau^2) splitting error; since the energy is a
// variational Rayleigh quotient, the converged energy error is O(dtau^4)
// and improves monotonically under step and grid refinement.
package groundstate

import (
	"math"

	"github.com/guoyyit/quantum-sim-go/internal/fft"
	"github.com/guoyyit/quantum-sim-go/internal/grid"
	"github.com/guoyyit/quantum-sim-go/internal/observables"
	"github.com/guoyyit/quantum-sim-go/internal/potential"
	"github.com/guoyyit/quantum-sim-go/internal/wavefunc"
)

// Params controls imaginary-time evolution.
type Params struct {
	DTau    float64 // imaginary time step
	MaxIter int     // maximum iterations
	Tol     float64 // convergence tolerance on |dE| (default 1e-12)
	Mass    float64 // particle mass (default 1)
	HBar    float64 // reduced Planck constant (default 1)
	Workers int     // parallel workers (default 1)
}

// Result reports the converged ground state and diagnostics.
type Result struct {
	Energy     float64                // converged ground-state energy estimate
	Psi        *wavefunc.WaveFunction // normalized ground-state wavefunction
	Iterations int                    // iterations performed
	Converged  bool                   // whether |dE| fell below Tol
	Residuals  []float64              // |E_k - E_{k-1}| per iteration (residual curve)
	Order      float64                // convergence order estimated from the residual tail
}

// Find propagates init in imaginary time until the energy change per
// iteration drops below Tol or MaxIter is reached. The initial state must
// have nonzero overlap with the true ground state.
func Find(g *grid.Grid, pot *potential.Potential, p Params, init *wavefunc.WaveFunction) *Result {
	if p.DTau <= 0 {
		panic("groundstate: DTau must be positive")
	}
	if p.Mass == 0 {
		p.Mass = 1
	}
	if p.HBar == 0 {
		p.HBar = 1
	}
	if p.Tol == 0 {
		p.Tol = 1e-12
	}
	if p.MaxIter <= 0 {
		p.MaxIter = 10000
	}
	if p.Workers <= 0 {
		p.Workers = 1
	}
	n := g.Nx * g.Ny
	halfV := make([]float64, n)
	for i, v := range pot.V {
		halfV[i] = math.Exp(-v * p.DTau / (2 * p.HBar))
	}
	kin := make([]float64, n)
	for i := 0; i < g.Nx; i++ {
		kx := g.Kx[i]
		for j := 0; j < g.Ny; j++ {
			ky := g.Ky[j]
			kin[i*g.Ny+j] = math.Exp(-p.HBar * (kx*kx + ky*ky) * p.DTau / (2 * p.Mass))
		}
	}

	psi := init.Clone()
	psi.Normalize(g.Dx, g.Dy)
	res := &Result{Residuals: make([]float64, 0, 256)}
	prevE := math.Inf(1)
	for iter := 1; iter <= p.MaxIter; iter++ {
		d := psi.Data
		for i := range d {
			d[i] *= complex(halfV[i], 0)
		}
		fft.FFT2D(d, g.Nx, g.Ny, false, p.Workers)
		for i := range d {
			d[i] *= complex(kin[i], 0)
		}
		fft.FFT2D(d, g.Nx, g.Ny, true, p.Workers)
		for i := range d {
			d[i] *= complex(halfV[i], 0)
		}
		psi.Normalize(g.Dx, g.Dy)

		e := observables.Energy(psi, g, pot, p.Mass, p.HBar)
		r := math.Abs(e - prevE)
		res.Residuals = append(res.Residuals, r)
		res.Iterations = iter
		prevE = e
		if r < p.Tol {
			res.Converged = true
			break
		}
	}
	res.Energy = prevE
	res.Psi = psi
	res.Order = estimateOrder(res.Residuals)
	return res
}

// estimateOrder derives the asymptotic convergence order of a residual
// sequence r_k via the standard three-term estimate
//
//	p = ln(r_{k-2}/r_{k-1}) / ln(r_{k-1}/r_k)
//
// averaged over the tail. Geometric (linear) convergence gives p ~ 1.
func estimateOrder(r []float64) float64 {
	var sum, cnt float64
	// Walk the tail backwards but skip residuals at the roundoff noise
	// floor, where the three-term estimate is meaningless.
	const floor = 1e-11
	for k := len(r) - 1; k >= 2 && cnt < 5; k-- {
		a, b, c := r[k-2], r[k-1], r[k]
		if a < floor || b < floor || c < floor {
			continue
		}
		if b != a && c != b {
			num := math.Log(a / b)
			den := math.Log(b / c)
			if den != 0 {
				sum += num / den
				cnt++
			}
		}
	}
	if cnt == 0 {
		return 0
	}
	return sum / cnt
}

// EstimateOrder fits err(h) ~ C*h^p by log-log least squares over the
// given step sizes and corresponding errors. Use it to measure the order
// of the converged energy with respect to the imaginary-time step or the
// grid spacing.
func EstimateOrder(hs, errs []float64) float64 {
	n := len(hs)
	if n != len(errs) || n < 2 {
		return 0
	}
	var sx, sy, sxx, sxy float64
	for i := 0; i < n; i++ {
		x := math.Log(hs[i])
		y := math.Log(errs[i])
		sx += x
		sy += y
		sxx += x * x
		sxy += x * y
	}
	den := float64(n)*sxx - sx*sx
	if den == 0 {
		return 0
	}
	return (float64(n)*sxy - sx*sy) / den
}
