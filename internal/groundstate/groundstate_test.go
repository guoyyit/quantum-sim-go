package groundstate_test

import (
	"math"
	"testing"

	"github.com/guoyyit/quantum-sim-go/internal/grid"
	"github.com/guoyyit/quantum-sim-go/internal/groundstate"
	"github.com/guoyyit/quantum-sim-go/internal/potential"
	"github.com/guoyyit/quantum-sim-go/internal/wavefunc"
)

// TestGroundStateHarmonic checks the imaginary-time solver against the
// analytic 2D harmonic oscillator ground state: E0 = (wx+wy)/2 and
// phi0(x,y) = pi^(-1/2) exp(-(x^2+y^2)/2) for hbar = m = w = 1.
func TestGroundStateHarmonic(t *testing.T) {
	g := grid.New(64, 64, 24, 24)
	pot := potential.Harmonic(g, 1, 1)
	init := wavefunc.FromFunc(g, func(x, y float64) complex128 {
		return complex(math.Exp(-(x*x+y*y)/8), 0)
	})
	init.Normalize(g.Dx, g.Dy)
	res := groundstate.Find(g, pot, groundstate.Params{DTau: 0.01, MaxIter: 20000, Tol: 1e-13}, init)
	if !res.Converged {
		t.Fatalf("did not converge after %d iterations", res.Iterations)
	}
	if d := math.Abs(res.Energy - 1.0); d > 1e-6 {
		t.Fatalf("E0 = %v, want 1.0 (diff %g)", res.Energy, d)
	}
	// Wavefunction matches the analytic ground state (real, positive).
	var num, den float64
	for i := 0; i < g.Nx; i++ {
		for j := 0; j < g.Ny; j++ {
			ex := math.Pow(math.Pi, -0.5) * math.Exp(-(g.X[i]*g.X[i]+g.Y[j]*g.Y[j])/2)
			d := real(res.Psi.At(i, j)) - ex
			num += d * d
			den += ex * ex
		}
	}
	if e := math.Sqrt(num / den); e > 1e-6 {
		t.Fatalf("ground state wavefunction relative L2 error = %g", e)
	}
	// Residual curve must decrease overall and end below the tolerance.
	r := res.Residuals
	if len(r) < 10 || r[len(r)-1] >= r[0] {
		t.Fatalf("residual curve does not decrease: first=%g last=%g len=%d", r[0], r[len(r)-1], len(r))
	}
	if r[len(r)-1] > 1e-13 {
		t.Fatalf("final residual %g above tolerance", r[len(r)-1])
	}
	// Imaginary-time convergence is geometric: order estimate ~ 1.
	if res.Order < 0.8 || res.Order > 1.2 {
		t.Fatalf("estimated order = %v, want ~1", res.Order)
	}
}

// TestGroundStateFromArbitraryInitial checks convergence to the same
// ground state from a non-Gaussian initial distribution.
func TestGroundStateFromArbitraryInitial(t *testing.T) {
	g := grid.New(64, 64, 24, 24)
	pot := potential.Harmonic(g, 1, 1)
	init := wavefunc.FromFunc(g, func(x, y float64) complex128 {
		return complex(1+0.3*math.Sin(x)*math.Cos(2*y)+0.2*math.Cos(3*x-y), 0)
	})
	init.Normalize(g.Dx, g.Dy)
	res := groundstate.Find(g, pot, groundstate.Params{DTau: 0.01, MaxIter: 40000, Tol: 1e-13}, init)
	if !res.Converged {
		t.Fatalf("did not converge after %d iterations", res.Iterations)
	}
	if d := math.Abs(res.Energy - 1.0); d > 1e-6 {
		t.Fatalf("E0 = %v, want 1.0", res.Energy)
	}
}

// TestGroundStateStepRefinement checks that the converged energy error
// decreases monotonically under step and grid refinement, and estimates
// the convergence order. The converged state carries the O(dtau^2)
// Strang splitting error, but because the energy is a variational
// Rayleigh quotient (stationary at the eigenstate), the energy error is
// O(dtau^4). An anharmonic potential is used so no commutator
// cancellation hides the error; the reference energy is computed with a
// very small step.
func TestGroundStateStepRefinement(t *testing.T) {
	anh := func(g *grid.Grid) *potential.Potential {
		return potential.FromFunc(g, func(x, y float64) float64 {
			return 0.5*(x*x+y*y) + 0.25*x*x*x*x
		})
	}
	newInit := func(g *grid.Grid) *wavefunc.WaveFunction {
		w := wavefunc.FromFunc(g, func(x, y float64) complex128 {
			return complex(math.Exp(-(x*x+y*y)/8), 0)
		})
		w.Normalize(g.Dx, g.Dy)
		return w
	}
	// Reference energy: converged with a very small step.
	gRef := grid.New(64, 64, 24, 24)
	ref := groundstate.Find(gRef, anh(gRef), groundstate.Params{DTau: 0.005, MaxIter: 80000, Tol: 1e-14}, newInit(gRef))
	if !ref.Converged {
		t.Fatalf("reference run did not converge")
	}
	exact := ref.Energy

	// Refine the imaginary-time step on a fixed grid.
	var dtaus, errs []float64
	for _, dtau := range []float64{0.08, 0.04, 0.02} {
		g := grid.New(64, 64, 24, 24)
		res := groundstate.Find(g, anh(g), groundstate.Params{DTau: dtau, MaxIter: 80000, Tol: 1e-14}, newInit(g))
		if !res.Converged {
			t.Fatalf("dtau=%g: did not converge", dtau)
		}
		dtaus = append(dtaus, dtau)
		errs = append(errs, math.Abs(res.Energy-exact))
	}
	for i := 0; i+1 < len(errs); i++ {
		if errs[i+1] >= errs[i] {
			t.Fatalf("energy error did not decrease under step refinement: %v", errs)
		}
	}
	if order := groundstate.EstimateOrder(dtaus, errs); order < 3.5 || order > 4.5 {
		t.Fatalf("estimated order = %v, want ~4 (variational energy error, errs %v)", order, errs)
	}

	// Refine the grid at a fixed small step on the harmonic potential
	// (analytic E0 = 1): error must not grow.
	var prev float64 = math.Inf(1)
	for _, n := range []int{32, 64, 128} {
		g := grid.New(n, n, 24, 24)
		pot := potential.Harmonic(g, 1, 1)
		res := groundstate.Find(g, pot, groundstate.Params{DTau: 0.01, MaxIter: 40000, Tol: 1e-14}, newInit(g))
		if !res.Converged {
			t.Fatalf("n=%d: did not converge", n)
		}
		e := math.Abs(res.Energy - 1.0)
		if e > prev {
			t.Fatalf("n=%d: error %g increased (previous %g)", n, e, prev)
		}
		prev = e
	}
}
