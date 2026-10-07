package sim

import (
	"math"
	"testing"
)

// l2Err returns the L2 error of the real and imaginary parts of a vs b.
func l2Err(a, b *Field) (re, im float64) {
	var sr, si float64
	for i := range a.Data {
		dr := real(a.Data[i]) - real(b.Data[i])
		di := imag(a.Data[i]) - imag(b.Data[i])
		sr += dr * dr
		si += di * di
	}
	n := float64(len(a.Data))
	return math.Sqrt(sr / n), math.Sqrt(si / n)
}

func freeCfg(nx int, dt float64, steps int) Config {
	return Config{
		Nx: nx, Ny: nx, Lx: 60, Ly: 60,
		Mass: 1, Hbar: 1, Dt: dt, Steps: steps,
		Workers: 1,
	}
}

// A free Gaussian packet must spread at the analytically known rate
// sigma(t) = sigma0 sqrt(1 + (hbar t / (2 m sigma0^2))^2).
func TestFreePacketSpreading(t *testing.T) {
	cfg := freeCfg(64, 0.005, 400) // T = 2
	sigma0 := 1.5
	solver, err := NewSolver(cfg, ZeroPotential(NewGrid(cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly)))
	if err != nil {
		t.Fatal(err)
	}
	g := solver.Grid()
	psi := GaussianPacket(g, 0, 0, sigma0, 0, 0)
	for i := 0; i < cfg.Steps; i++ {
		solver.Step(psi)
	}
	T := cfg.Dt * float64(cfg.Steps)
	want := sigma0 * math.Sqrt(1+math.Pow(cfg.Hbar*T/(2*cfg.Mass*sigma0*sigma0), 2))
	got := WidthX(psi, g)
	if rel := math.Abs(got-want) / want; rel > 2e-3 {
		t.Fatalf("width: got %g want %g (rel err %g)", got, want, rel)
	}
	if d := math.Abs(Norm(psi, g.CellArea()) - 1); d > 1e-12 {
		t.Fatalf("norm drifted by %g", d)
	}
}

// Real- and imaginary-part errors must shrink as dt is refined, at the
// second-order rate of the Strang splitting. A harmonic potential is used
// because the free-particle kinetic propagator is exact in time.
func TestErrorDecreasesWithStepRefinement(t *testing.T) {
	T := 1.0
	ref := evolveHarmonic(t, 64, T/4096, 4096)
	var prevRe, prevIm float64
	for _, steps := range []int{128, 256, 512} {
		got := evolveHarmonic(t, 64, T/float64(steps), steps)
		re, im := l2Err(got, ref)
		t.Logf("steps=%d errRe=%.3e errIm=%.3e", steps, re, im)
		if steps > 128 {
			if re >= prevRe || im >= prevIm {
				t.Fatalf("error did not decrease: (%g,%g) -> (%g,%g)", prevRe, prevIm, re, im)
			}
			if r := prevRe / re; r < 2 || r > 8 {
				t.Fatalf("real-part error ratio %g, expected ~4 (2nd order)", r)
			}
		}
		prevRe, prevIm = re, im
	}
}

// Spatial (spectral) error must decrease under grid refinement, measured
// against the finest grid (self-convergence; the periodized packet differs
// from the infinite-line analytic value by an N-independent amount).
func TestErrorDecreasesWithGridRefinement(t *testing.T) {
	// sigma << L so the packet vanishes at the periodic seam; otherwise the
	// seam discontinuity dominates and converges only algebraically.
	sigma := 1.5
	kineticOn := func(n int) float64 {
		cfg := Config{Nx: n, Ny: n, Lx: 30, Ly: 30, Mass: 1, Hbar: 1, Dt: 0.01, Steps: 1, Workers: 1}
		g := NewGrid(n, n, cfg.Lx, cfg.Ly)
		solver, err := NewSolver(cfg, ZeroPotential(g))
		if err != nil {
			t.Fatal(err)
		}
		return solver.KineticExpectation(GaussianPacket(g, 0, 0, sigma, 0, 0))
	}
	ref := kineticOn(256)
	var prev float64 = math.Inf(1)
	for _, n := range []int{16, 32, 64} {
		errT := math.Abs(kineticOn(n) - ref)
		t.Logf("N=%d kinetic error vs N=256: %.3e", n, errT)
		if errT >= prev {
			t.Fatalf("kinetic error did not decrease: %g -> %g", prev, errT)
		}
		prev = errT
	}
	if prev > 1e-10 {
		t.Fatalf("finest grid kinetic error %g too large", prev)
	}
}

// Long runs must remain bounded: norm and energy drift stay small and do
// not grow without control as the step count increases.
func TestLongRunStability(t *testing.T) {
	cfg := freeCfg(32, 0.02, 20000)
	g := NewGrid(cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly)
	solver, err := NewSolver(cfg, ZeroPotential(g))
	if err != nil {
		t.Fatal(err)
	}
	psi := GaussianPacket(g, 0, 0, 2, 1, 0)
	e0 := solver.EnergyExpectation(psi)
	var midNormDrift, midEDrift float64
	for i := 1; i <= cfg.Steps; i++ {
		solver.Step(psi)
		if i == cfg.Steps/2 {
			midNormDrift = math.Abs(Norm(psi, g.CellArea()) - 1)
			midEDrift = math.Abs(solver.EnergyExpectation(psi)-e0) / e0
		}
	}
	normDrift := math.Abs(Norm(psi, g.CellArea()) - 1)
	eDrift := math.Abs(solver.EnergyExpectation(psi)-e0) / e0
	t.Logf("mid: normDrift=%.3e eDrift=%.3e; final: normDrift=%.3e eDrift=%.3e",
		midNormDrift, midEDrift, normDrift, eDrift)
	if normDrift > 1e-9 {
		t.Fatalf("norm drift %g out of control", normDrift)
	}
	if eDrift > 1e-6 {
		t.Fatalf("energy drift %g out of control", eDrift)
	}
	// Drift must not blow up between the midpoint and the end.
	if normDrift > 100*(midNormDrift+1e-15) {
		t.Fatalf("norm drift growing: mid %g final %g", midNormDrift, normDrift)
	}
	if eDrift > 100*(midEDrift+1e-15) {
		t.Fatalf("energy drift growing: mid %g final %g", midEDrift, eDrift)
	}
}

// evolveHarmonic evolves a displaced Gaussian in a 2D harmonic trap.
func evolveHarmonic(t *testing.T, n int, dt float64, steps int) *Field {
	t.Helper()
	cfg := Config{Nx: n, Ny: n, Lx: 20, Ly: 20, Mass: 1, Hbar: 1, Dt: dt, Steps: steps, Workers: 1}
	g := NewGrid(cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly)
	solver, err := NewSolver(cfg, HarmonicPotential(g, 1, 1, 0, 0))
	if err != nil {
		t.Fatal(err)
	}
	psi := GaussianPacket(g, 2, 0, 1, 0, 0)
	for i := 0; i < steps; i++ {
		solver.Step(psi)
	}
	return psi
}
