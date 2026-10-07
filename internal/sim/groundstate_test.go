package sim

import (
	"math"
	"testing"
)

// groundEnergy runs imaginary-time evolution in a 2D harmonic trap and
// returns the converged energy. Exact ground-state energy: hbar*omega.
func groundEnergy(t *testing.T, n int, dt float64) (float64, *GroundStateResult) {
	t.Helper()
	cfg := Config{
		Nx: n, Ny: n, Lx: 16, Ly: 16,
		Mass: 1, Hbar: 1, Dt: dt,
		Workers: 1,
	}
	g := NewGrid(n, n, cfg.Lx, cfg.Ly)
	V := HarmonicPotential(g, cfg.Mass, 1, 0, 0)
	// Arbitrary non-symmetric initial state.
	psi0 := GaussianPacket(g, 1.2, -0.7, 1.3, 0.3, 0)
	res, err := GroundState(cfg, V, psi0, 1e-12, 200000)
	if err != nil {
		t.Fatal(err)
	}
	return res.Energy, res
}

func TestGroundStateEnergy(t *testing.T) {
	e, res := groundEnergy(t, 64, 0.02)
	if !res.Converged {
		t.Fatalf("did not converge in %d steps", res.Steps)
	}
	if d := math.Abs(e - 1.0); d > 1e-6 {
		t.Fatalf("ground energy %g, want 1.0 (hbar*omega), err %g", e, d)
	}
	if n := Norm(res.Psi, NewGrid(64, 64, 16, 16).CellArea()); math.Abs(n-1) > 1e-12 {
		t.Fatalf("ground state not normalized: %g", n)
	}
}

// The residual curve must decay (geometrically in imaginary time) and the
// converged energy must improve monotonically with dt refinement, at
// second order in dt.
func TestGroundStateConvergenceOrder(t *testing.T) {
	dts := []float64{0.08, 0.04, 0.02}
	var energies []float64
	for _, dt := range dts {
		e, res := groundEnergy(t, 64, dt)
		if !res.Converged {
			t.Fatalf("dt=%g did not converge", dt)
		}
		energies = append(energies, e)
		// Residual curve: strictly recorded per step, decaying overall.
		if len(res.Residuals) != res.Steps {
			t.Fatalf("residual curve incomplete: %d entries for %d steps", len(res.Residuals), res.Steps)
		}
		first := res.Residuals[0]
		last := res.Residuals[len(res.Residuals)-1]
		if last > 1e-3*first {
			t.Fatalf("dt=%g residual did not decay: first %g last %g", dt, first, last)
		}
	}
	var prevErr float64 = math.Inf(1)
	for i, dt := range dts {
		err := math.Abs(energies[i] - 1.0)
		t.Logf("dt=%g E=%.10f err=%.3e", dt, energies[i], err)
		if err >= prevErr {
			t.Fatalf("energy error did not decrease with dt refinement: %g -> %g", prevErr, err)
		}
		prevErr = err
	}
	// The state converges at O(dt^2); the energy, being stationary at the
	// eigenstate, converges at O(dt^4) (variational superconvergence).
	if p := EstimateOrder(energies[0], energies[1], energies[2]); p < 1.8 {
		t.Fatalf("convergence order %g, want >= 2", p)
	}
}

// Ground-state energy error must also decrease under grid refinement.
func TestGroundStateGridRefinement(t *testing.T) {
	var prevErr float64 = math.Inf(1)
	for _, n := range []int{8, 16, 32} {
		e, _ := groundEnergy(t, n, 0.002)
		err := math.Abs(e - 1.0)
		t.Logf("N=%d E=%.10f err=%.3e", n, e, err)
		if err >= prevErr {
			t.Fatalf("energy error did not decrease with grid refinement: %g -> %g", prevErr, err)
		}
		prevErr = err
	}
}

// The ground state must be stationary under real-time evolution: |psi|^2
// stays put and the phase rotates at E0/hbar.
func TestGroundStateStationarity(t *testing.T) {
	cfg := Config{
		Nx: 64, Ny: 64, Lx: 16, Ly: 16,
		Mass: 1, Hbar: 1, Dt: 0.005, Steps: 400,
		Workers: 1,
	}
	g := NewGrid(64, 64, 16, 16)
	V := HarmonicPotential(g, 1, 1, 0, 0)
	_, gs := groundEnergy(t, 64, 0.005)

	solver, err := NewSolver(cfg, V)
	if err != nil {
		t.Fatal(err)
	}
	psi := gs.Psi.Clone()
	before := psi.Clone()
	for i := 0; i < cfg.Steps; i++ {
		solver.Step(psi)
	}
	// Density unchanged.
	for i := range psi.Data {
		d := abs2(psi.Data[i]) - abs2(before.Data[i])
		if math.Abs(d) > 1e-5 {
			t.Fatalf("density changed at %d by %g", i, d)
		}
	}
	// Phase rotated by E0*T/hbar at the origin.
	T := cfg.Dt * float64(cfg.Steps)
	wantPhase := -gs.Energy * T / cfg.Hbar
	gotPhase := phase(psi.At(32, 32)) - phase(before.At(32, 32))
	if d := angDiff(gotPhase, wantPhase); math.Abs(d) > 1e-3 {
		t.Fatalf("phase rotation %g, want %g (diff %g)", gotPhase, wantPhase, d)
	}
}

func abs2(c complex128) float64 { return real(c)*real(c) + imag(c)*imag(c) }

func phase(c complex128) float64 { return math.Atan2(imag(c), real(c)) }

func angDiff(a, b float64) float64 {
	d := a - b
	for d > math.Pi {
		d -= 2 * math.Pi
	}
	for d < -math.Pi {
		d += 2 * math.Pi
	}
	return d
}
