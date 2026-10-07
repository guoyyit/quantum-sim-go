package solver_test

import (
	"math"
	"math/cmplx"
	"testing"

	"github.com/guoyyit/quantum-sim-go/internal/grid"
	"github.com/guoyyit/quantum-sim-go/internal/observables"
	"github.com/guoyyit/quantum-sim-go/internal/potential"
	"github.com/guoyyit/quantum-sim-go/internal/solver"
	"github.com/guoyyit/quantum-sim-go/internal/wavefunc"
)

// gaussian builds a normalized 2D Gaussian wavepacket with |psi|^2
// variances sx^2, sy^2, centered at (x0,y0) with plane-wave phase (kx,ky).
func gaussian(g *grid.Grid, x0, y0, sx, sy, kx, ky float64) *wavefunc.WaveFunction {
	w := wavefunc.FromFunc(g, func(x, y float64) complex128 {
		dx, dy := x-x0, y-y0
		env := math.Exp(-dx*dx/(4*sx*sx) - dy*dy/(4*sy*sy))
		return complex(env*math.Cos(kx*x+ky*y), env*math.Sin(kx*x+ky*y))
	})
	w.Normalize(g.Dx, g.Dy)
	return w
}

// freeGaussian returns the analytic free-particle evolution of a 1D
// Gaussian with initial variance s^2, center x0 and wavenumber k.
func freeGaussian(x, t, x0, s, k float64) complex128 {
	tau := t / (2 * s * s) // hbar = m = 1
	gam := complex(1, tau)
	norm := math.Pow(2*math.Pi*s*s, -0.25)
	xc := x0 + k*t
	env := cmplx.Exp(-complex(x-xc, 0) * complex(x-xc, 0) / (complex(4*s*s, 0) * gam))
	phase := cmplx.Exp(complex(0, k*x-k*k*t/2))
	return complex(norm, 0) * cmplx.Pow(gam, complex(-0.5, 0)) * env * phase
}

func meanX(w *wavefunc.WaveFunction, g *grid.Grid) float64 {
	var s float64
	for i := 0; i < g.Nx; i++ {
		for j := 0; j < g.Ny; j++ {
			v := w.At(i, j)
			s += g.X[i] * (real(v)*real(v) + imag(v)*imag(v))
		}
	}
	return s * g.Dx * g.Dy
}

func varX(w *wavefunc.WaveFunction, g *grid.Grid) float64 {
	mx := meanX(w, g)
	var s float64
	for i := 0; i < g.Nx; i++ {
		for j := 0; j < g.Ny; j++ {
			v := w.At(i, j)
			d := g.X[i] - mx
			s += d * d * (real(v)*real(v) + imag(v)*imag(v))
		}
	}
	return s * g.Dx * g.Dy
}

func l2Error(w *wavefunc.WaveFunction, exact *wavefunc.WaveFunction, g *grid.Grid) float64 {
	var num, den float64
	for i := range w.Data {
		d := w.Data[i] - exact.Data[i]
		num += real(d)*real(d) + imag(d)*imag(d)
		den += real(exact.Data[i])*real(exact.Data[i]) + imag(exact.Data[i])*imag(exact.Data[i])
	}
	return math.Sqrt(num / den)
}

func TestFreeWavepacketSpreading(t *testing.T) {
	g := grid.New(128, 128, 32, 32)
	pot := potential.Zero(g)
	const sx, kx, T = 1.0, 3.0, 1.0
	psi := gaussian(g, 0, 0, sx, sx, kx, 0)
	sv := solver.New(g, pot, solver.Params{Dt: 0.002, Steps: int(T / 0.002), Workers: 4})
	if st := sv.Run(psi, 0, solver.Hooks{}); st != solver.StatusOK {
		t.Fatalf("status = %v", st)
	}
	// Norm conserved by unitarity.
	if n := observables.Norm(psi, g); math.Abs(n-1) > 1e-10 {
		t.Fatalf("norm = %v, want 1", n)
	}
	// Mean position follows the group velocity.
	if mx := meanX(psi, g); math.Abs(mx-kx*T) > 1e-6 {
		t.Fatalf("<x> = %v, want %v", mx, kx*T)
	}
	// Analytic spreading: sigma^2(t) = sigma^2 (1 + (t/(2 sigma^2))^2).
	wantVar := sx * sx * (1 + T*T/(4*sx*sx*sx*sx))
	if v := varX(psi, g); math.Abs(v-wantVar)/wantVar > 1e-6 {
		t.Fatalf("var(x) = %v, want %v", v, wantVar)
	}
	// Full complex amplitude matches the analytic solution.
	exact := wavefunc.FromFunc(g, func(x, y float64) complex128 {
		return freeGaussian(x, T, 0, sx, kx) * freeGaussian(y, T, 0, sx, 0)
	})
	if err := l2Error(psi, exact, g); err > 1e-8 {
		t.Fatalf("relative L2 error vs analytic = %g", err)
	}
}

func TestHarmonicOscillatorDtConvergence(t *testing.T) {
	// Ehrenfest theorem: <x>(t) = x0 cos(omega t) exactly. The splitting
	// error is O(dt^2), so halving dt must quarter the error.
	const omega, x0, T = 1.0, 1.5, 1.0
	var errs []float64
	for _, dt := range []float64{0.02, 0.01, 0.005} {
		g := grid.New(64, 64, 24, 24)
		pot := potential.Harmonic(g, omega, omega)
		s := math.Sqrt(0.5) // coherent-state width sqrt(hbar/(2 m omega))
		psi := gaussian(g, x0, 0, s, s, 0, 0)
		sv := solver.New(g, pot, solver.Params{Dt: dt, Steps: int(T/dt + 0.5), Workers: 4})
		if st := sv.Run(psi, 0, solver.Hooks{}); st != solver.StatusOK {
			t.Fatalf("dt=%g: status = %v", dt, st)
		}
		errs = append(errs, math.Abs(meanX(psi, g)-x0*math.Cos(omega*T)))
	}
	for i := 0; i+1 < len(errs); i++ {
		if errs[i+1] >= errs[i] {
			t.Fatalf("error did not decrease under refinement: %v", errs)
		}
		ratio := errs[i] / errs[i+1]
		if ratio < 3.2 || ratio > 5.5 {
			t.Fatalf("expected ~4x error reduction (2nd order), got %gx (errs %v)", ratio, errs)
		}
	}
}

func TestLongTimeStability(t *testing.T) {
	// Energy drift must stay bounded over long runs (no secular growth).
	g := grid.New(32, 32, 24, 24)
	pot := potential.Harmonic(g, 1, 1)
	psi := gaussian(g, 1.5, 0, math.Sqrt(0.5), math.Sqrt(0.5), 0, 0)
	sv := solver.New(g, pot, solver.Params{
		Dt: 0.02, Steps: 10000, SampleEvery: 500, Workers: 4,
		NormTol: 1e-9, EnergyTol: 1e-3,
	})
	st := sv.Run(psi, 0, solver.Hooks{})
	if st != solver.StatusOK {
		t.Fatalf("status = %v (norm drift %g, energy drift %g)",
			st, sv.MaxNormDrift, sv.MaxEnergyDrift)
	}
	if sv.MaxNormDrift > 1e-9 {
		t.Fatalf("norm drift %g exceeds 1e-9", sv.MaxNormDrift)
	}
	if sv.MaxEnergyDrift > 1e-3 {
		t.Fatalf("energy drift %g exceeds 1e-3", sv.MaxEnergyDrift)
	}
}

// hoEigen returns the 1D harmonic-oscillator eigenstate n (hbar=m=omega=1).
func hoEigen(x float64, n int) float64 {
	switch n {
	case 0:
		return math.Pow(math.Pi, -0.25) * math.Exp(-x*x/2)
	default:
		return math.Sqrt2 * math.Pow(math.Pi, -0.25) * x * math.Exp(-x*x/2)
	}
}

func TestSuperpositionPhaseAndInterference(t *testing.T) {
	// psi = (phi_0(x)phi_0(y) + phi_1(x)phi_0(y))/sqrt(2).
	// The relative phase evolves as exp(-i (E1-E0) t) = exp(-i t), and
	// <x>(t) = cos(t)/sqrt(2) exactly (hbar = m = omega = 1).
	g := grid.New(64, 64, 24, 24)
	pot := potential.Harmonic(g, 1, 1)
	phi := wavefunc.FromFunc(g, func(x, y float64) complex128 {
		return complex(hoEigen(x, 0)*hoEigen(y, 0)+hoEigen(x, 1)*hoEigen(y, 0), 0)
	})
	phi.Normalize(g.Dx, g.Dy)

	// Reference eigenstate products for projection.
	ref0 := wavefunc.FromFunc(g, func(x, y float64) complex128 {
		return complex(hoEigen(x, 0)*hoEigen(y, 0), 0)
	})
	ref0.Normalize(g.Dx, g.Dy)
	ref1 := wavefunc.FromFunc(g, func(x, y float64) complex128 {
		return complex(hoEigen(x, 1)*hoEigen(y, 0), 0)
	})
	ref1.Normalize(g.Dx, g.Dy)

	const dt = 0.005
	const stepsPerSample = 40 // 0.2 time units
	sv := solver.New(g, pot, solver.Params{Dt: dt, Steps: 0, Workers: 4})
	psi := phi
	for k := 1; k <= 30; k++ {
		sv2 := solver.New(g, pot, solver.Params{Dt: dt, Steps: stepsPerSample, Workers: 4})
		_ = sv
		if st := sv2.Run(psi, 0, solver.Hooks{}); st != solver.StatusOK {
			t.Fatalf("status = %v", st)
		}
		tt := float64(k*stepsPerSample) * dt
		// <x>(t) = cos(t)/sqrt(2).
		if mx := meanX(psi, g); math.Abs(mx-math.Cos(tt)/math.Sqrt2) > 2e-3 {
			t.Fatalf("t=%g: <x> = %v, want cos(t)/sqrt(2) = %v", tt, mx, math.Cos(tt)/math.Sqrt2)
		}
		// Relative phase of the two amplitudes: c1/c0 = exp(-i t).
		c0 := wavefunc.Inner(ref0, psi, g.Dx, g.Dy)
		c1 := wavefunc.Inner(ref1, psi, g.Dx, g.Dy)
		ratio := c1 / c0
		want := cmplx.Exp(complex(0, -tt))
		if d := cmplx.Abs(ratio - want); d > 2e-3 {
			t.Fatalf("t=%g: c1/c0 = %v, want %v (diff %g)", tt, ratio, want, d)
		}
	}
}

func TestParallelConsistency(t *testing.T) {
	g := grid.New(64, 64, 24, 24)
	pot := potential.Harmonic(g, 1, 1)
	run := func(workers int) ([]complex128, []solver.Frame) {
		psi := gaussian(g, 1.5, -0.5, math.Sqrt(0.5), math.Sqrt(0.5), 0.3, 0)
		sv := solver.New(g, pot, solver.Params{
			Dt: 0.01, Steps: 200, SampleEvery: 50, Workers: workers,
		})
		var frames []solver.Frame
		st := sv.Run(psi, 0, solver.Hooks{Sample: func(f solver.Frame) {
			frames = append(frames, f)
		}})
		if st != solver.StatusOK {
			t.Fatalf("workers=%d: status = %v", workers, st)
		}
		return psi.Data, frames
	}
	baseData, baseFrames := run(1)
	for _, w := range []int{2, 3, 4, 8} {
		data, frames := run(w)
		if len(frames) != len(baseFrames) {
			t.Fatalf("workers=%d: got %d frames, want %d", w, len(frames), len(baseFrames))
		}
		for i := range data {
			if data[i] != baseData[i] {
				t.Fatalf("workers=%d: final state differs at %d", w, i)
			}
		}
		for fi := range frames {
			if frames[fi].Step != baseFrames[fi].Step || frames[fi].Time != baseFrames[fi].Time {
				t.Fatalf("workers=%d: frame %d metadata differs", w, fi)
			}
			for i := range frames[fi].Psi.Data {
				if frames[fi].Psi.Data[i] != baseFrames[fi].Psi.Data[i] {
					t.Fatalf("workers=%d: frame %d differs at element %d", w, fi, i)
				}
			}
		}
	}
}

func TestDriftDiagnostics(t *testing.T) {
	g := grid.New(32, 32, 24, 24)
	pot := potential.Harmonic(g, 1, 1)

	// Absurdly tight norm tolerance must surface StatusNormDrift.
	psi := gaussian(g, 1, 0, math.Sqrt(0.5), math.Sqrt(0.5), 0, 0)
	sv := solver.New(g, pot, solver.Params{Dt: 0.01, Steps: 50, NormTol: 1e-30})
	if st := sv.Run(psi, 0, solver.Hooks{}); st != solver.StatusNormDrift {
		t.Fatalf("status = %v, want norm-drift", st)
	}

	// Absurdly tight energy tolerance must surface StatusEnergyDrift.
	psi = gaussian(g, 1, 0, math.Sqrt(0.5), math.Sqrt(0.5), 0, 0)
	sv = solver.New(g, pot, solver.Params{Dt: 0.01, Steps: 50, EnergyTol: 1e-30})
	if st := sv.Run(psi, 0, solver.Hooks{}); st != solver.StatusEnergyDrift {
		t.Fatalf("status = %v, want energy-drift", st)
	}

	// The two statuses are distinguishable.
	if solver.StatusNormDrift.String() == solver.StatusEnergyDrift.String() ||
		solver.StatusOK.String() == solver.StatusNormDrift.String() {
		t.Fatalf("status strings are not distinguishable")
	}
}

func TestFreeWavepacketGridRefinement(t *testing.T) {
	// The spectral spatial discretization must drive the complex-amplitude
	// error down as the grid refines (here: momentum-aliasing limited).
	const sx, kx, T = 1.0, 3.0, 1.0
	var prev float64 = math.Inf(1)
	for _, n := range []int{64, 128} {
		g := grid.New(n, n, 32, 32)
		pot := potential.Zero(g)
		psi := gaussian(g, 0, 0, sx, sx, kx, 0)
		sv := solver.New(g, pot, solver.Params{Dt: 0.002, Steps: int(T / 0.002), Workers: 4})
		if st := sv.Run(psi, 0, solver.Hooks{}); st != solver.StatusOK {
			t.Fatalf("n=%d: status = %v", n, st)
		}
		exact := wavefunc.FromFunc(g, func(x, y float64) complex128 {
			return freeGaussian(x, T, 0, sx, kx) * freeGaussian(y, T, 0, sx, 0)
		})
		err := l2Error(psi, exact, g)
		if err >= prev {
			t.Fatalf("n=%d: error %g did not decrease (previous %g)", n, err, prev)
		}
		prev = err
	}
}
