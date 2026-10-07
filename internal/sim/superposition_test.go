package sim

import (
	"math"
	"testing"
)

// A superposition of two plane waves produces a moving interference
// fringe: psi = (e^{i(k1 x - w1 t)} + e^{i(k2 x - w2 t)}) / sqrt(2 Lx Ly).
// The spectral method integrates commensurate plane waves exactly, so the
// full complex amplitude must match the analytic solution to roundoff.
func TestPlaneWaveInterference(t *testing.T) {
	cfg := Config{
		Nx: 32, Ny: 32, Lx: 2 * math.Pi, Ly: 2 * math.Pi,
		Mass: 1, Hbar: 1, Dt: 0.001, Steps: 700,
		Workers: 1,
	}
	g := NewGrid(cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly)
	solver, err := NewSolver(cfg, ZeroPotential(g))
	if err != nil {
		t.Fatal(err)
	}
	k1, k2 := 1.0, 3.0
	psi := AddFields(PlaneWave(g, k1, 0), PlaneWave(g, k2, 0))
	Normalize(psi, g)

	T := cfg.Dt * float64(cfg.Steps)
	for i := 0; i < cfg.Steps; i++ {
		solver.Step(psi)
	}

	w1 := cfg.Hbar * k1 * k1 / (2 * cfg.Mass)
	w2 := cfg.Hbar * k2 * k2 / (2 * cfg.Mass)
	norm := 1 / math.Sqrt(2*g.Lx*g.Ly)
	var maxErr float64
	for iy := 0; iy < g.Ny; iy++ {
		for ix := 0; ix < g.Nx; ix++ {
			x := g.X(ix)
			s1, c1 := math.Sincos(k1*x - w1*T)
			s2, c2 := math.Sincos(k2*x - w2*T)
			want := complex(norm*(c1+c2), norm*(s1+s2))
			got := psi.At(ix, iy)
			dr := real(got) - real(want)
			di := imag(got) - imag(want)
			if d := math.Hypot(dr, di); d > maxErr {
				maxErr = d
			}
		}
	}
	if maxErr > 1e-11 {
		t.Fatalf("complex amplitude deviates from analytic by %g", maxErr)
	}
	// The density fringe must match the analytic interference pattern
	// pointwise and show deep contrast (sampled min/max depend on the grid).
	mean := 1 / (g.Lx * g.Ly)
	var dmin, dmax float64 = math.Inf(1), 0
	for iy := 0; iy < g.Ny; iy++ {
		for ix := 0; ix < g.Nx; ix++ {
			x := g.X(ix)
			wantD := mean * (1 + math.Cos((k1-k2)*x-(w1-w2)*T))
			gotD := abs2(psi.At(ix, iy))
			if d := math.Abs(gotD - wantD); d > 1e-10*mean {
				t.Fatalf("density at (%g,%g): got %g want %g", x, g.Y(iy), gotD, wantD)
			}
			if gotD < dmin {
				dmin = gotD
			}
			if gotD > dmax {
				dmax = gotD
			}
		}
	}
	if dmax/dmin < 100 {
		t.Fatalf("fringe contrast %g too shallow", dmax/dmin)
	}
}

// A superposition of the two lowest harmonic-oscillator states oscillates:
// <x>(t) = (1/sqrt(2)) cos(omega t) for hbar = m = omega = 1.
func TestSuperpositionOscillation(t *testing.T) {
	cfg := Config{
		Nx: 64, Ny: 64, Lx: 16, Ly: 16,
		Mass: 1, Hbar: 1, Dt: 0.005, Steps: 0,
		Workers: 1,
	}
	g := NewGrid(cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly)
	solver, err := NewSolver(cfg, HarmonicPotential(g, 1, 1, 0, 0))
	if err != nil {
		t.Fatal(err)
	}
	// Analytic eigenstates: psi0 = pi^-1/2 e^{-r^2/2}, psi1 = sqrt(2) x psi0.
	psi0 := NewField(g)
	psi1 := NewField(g)
	for iy := 0; iy < g.Ny; iy++ {
		for ix := 0; ix < g.Nx; ix++ {
			x, y := g.X(ix), g.Y(iy)
			v := math.Exp(-(x*x+y*y)/2) / math.Sqrt(math.Pi)
			psi0.Set(ix, iy, complex(v, 0))
			psi1.Set(ix, iy, complex(math.Sqrt2*x*v, 0))
		}
	}
	Normalize(psi0, g)
	Normalize(psi1, g)
	psi := AddFields(psi0, psi1)
	Normalize(psi, g)

	// Evolve one full classical period, sampling <x>.
	dt := cfg.Dt
	steps := int(2*math.Pi/dt + 0.5)
	sampleEvery := 40
	var maxErr float64
	for i := 1; i <= steps; i++ {
		solver.Step(psi)
		if i%sampleEvery == 0 {
			tnow := float64(i) * dt
			x, _ := PositionExpectation(psi, g)
			want := math.Cos(tnow) / math.Sqrt2
			if d := math.Abs(x - want); d > maxErr {
				maxErr = d
			}
		}
	}
	if maxErr > 2e-3 {
		t.Fatalf("<x>(t) deviates from cos(t)/sqrt(2) by %g", maxErr)
	}
	// After a full period the state must return to itself (up to a phase).
	x0, _ := PositionExpectation(psi, g)
	if d := math.Abs(x0 - 1/math.Sqrt2); d > 2e-3 {
		t.Fatalf("after one period <x>=%g, want %g", x0, 1/math.Sqrt2)
	}
}
