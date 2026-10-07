package sim

import (
	"errors"
	"math"
	"testing"
)

// The unitary integrator must conserve the norm to roundoff and keep the
// energy drift bounded (symplectic oscillation, O(dt^2) amplitude): the
// drift must not grow with step count and must shrink as dt is refined.
func TestConservationNormAndEnergy(t *testing.T) {
	maxDrifts := func(dt float64, steps int) (normDrift, eDrift float64) {
		cfg := Config{
			Nx: 64, Ny: 64, Lx: 20, Ly: 20,
			Mass: 1, Hbar: 1, Dt: dt, Steps: steps,
			Workers: 1,
		}
		g := NewGrid(cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly)
		V := HarmonicPotential(g, 1, 1, 0, 0)
		solver, err := NewSolver(cfg, V)
		if err != nil {
			t.Fatal(err)
		}
		psi := GaussianPacket(g, 2, 1, 1, 0.5, -0.5)
		n0 := Norm(psi, g.CellArea())
		e0 := solver.EnergyExpectation(psi)
		for i := 0; i < cfg.Steps; i++ {
			solver.Step(psi)
			if i%25 == 0 {
				normDrift = math.Max(normDrift, math.Abs(Norm(psi, g.CellArea())-n0))
				eDrift = math.Max(eDrift, math.Abs(solver.EnergyExpectation(psi)-e0)/e0)
			}
		}
		return normDrift, eDrift
	}

	// Same physical time T=5, two resolutions.
	nd1, ed1 := maxDrifts(0.01, 500)
	nd2, ed2 := maxDrifts(0.005, 1000)
	t.Logf("dt=0.01:  norm drift %.3e, energy drift %.3e", nd1, ed1)
	t.Logf("dt=0.005: norm drift %.3e, energy drift %.3e", nd2, ed2)
	if nd1 > 1e-12 || nd2 > 1e-12 {
		t.Fatalf("norm drift out of control: %g, %g", nd1, nd2)
	}
	if ed1 > 1e-3 {
		t.Fatalf("energy drift %g too large for dt=0.01", ed1)
	}
	// Bounded oscillation: halving dt must reduce the drift ~4x.
	if r := ed1 / ed2; r < 2 || r > 8 {
		t.Fatalf("energy drift ratio %g, expected ~4 (bounded O(dt^2))", r)
	}
	// Not growing with step count: doubling the run must not inflate it.
	_, edLong := maxDrifts(0.005, 2000)
	if edLong > 4*ed2 {
		t.Fatalf("energy drift grows with step count: %g -> %g", ed2, edLong)
	}
}

// Energy of a plane wave must equal hbar^2 k^2 / 2m exactly.
func TestEnergyExpectationPlaneWave(t *testing.T) {
	cfg := Config{
		Nx: 32, Ny: 32, Lx: 2 * math.Pi, Ly: 2 * math.Pi,
		Mass: 2, Hbar: 1, Dt: 0.01, Steps: 1,
		Workers: 1,
	}
	g := NewGrid(cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly)
	solver, err := NewSolver(cfg, ZeroPotential(g))
	if err != nil {
		t.Fatal(err)
	}
	kx, ky := 3.0, -2.0
	psi := PlaneWave(g, kx, ky)
	want := cfg.Hbar * cfg.Hbar * (kx*kx + ky*ky) / (2 * cfg.Mass)
	if got := solver.EnergyExpectation(psi); math.Abs(got-want) > 1e-12 {
		t.Fatalf("plane wave energy %g, want %g", got, want)
	}
}

// Drift beyond tolerance must stop the run with a distinguishable status.
func TestNormDriftDetected(t *testing.T) {
	cfg := Config{
		Nx: 32, Ny: 32, Lx: 20, Ly: 20,
		Mass: 1, Hbar: 1, Dt: 0.01, Steps: 200,
		Workers: 1, OutputDir: t.TempDir(),
		SnapshotEvery: 10,
		NormDriftTol:  1e-18, // impossibly tight: roundoff triggers it
	}
	g := NewGrid(cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly)
	res, err := Run(cfg, HarmonicPotential(g, 1, 1, 0, 0), GaussianPacket(g, 1, 0, 1, 0, 0))
	var de *DriftError
	if !errors.As(err, &de) {
		t.Fatalf("expected DriftError, got %v", err)
	}
	if de.Diag.Status != StatusNormDrift {
		t.Fatalf("status %s, want %s", de.Diag.Status, StatusNormDrift)
	}
	if res.Status != StatusNormDrift {
		t.Fatalf("result status %s, want %s", res.Status, StatusNormDrift)
	}
	if last := res.History[len(res.History)-1]; last.Step >= cfg.Steps {
		t.Fatalf("run did not stop early: last step %d of %d", last.Step, cfg.Steps)
	}
}

func TestEnergyDriftDetected(t *testing.T) {
	cfg := Config{
		Nx: 32, Ny: 32, Lx: 20, Ly: 20,
		Mass: 1, Hbar: 1, Dt: 0.01, Steps: 200,
		Workers: 1, OutputDir: t.TempDir(),
		SnapshotEvery:  10,
		EnergyDriftTol: 1e-18, // impossibly tight
	}
	g := NewGrid(cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly)
	res, err := Run(cfg, HarmonicPotential(g, 1, 1, 0, 0), GaussianPacket(g, 1, 0, 1, 0, 0))
	var de *DriftError
	if !errors.As(err, &de) {
		t.Fatalf("expected DriftError, got %v", err)
	}
	if de.Diag.Status != StatusEnergyDrift {
		t.Fatalf("status %s, want %s", de.Diag.Status, StatusEnergyDrift)
	}
	if res.Status != StatusEnergyDrift {
		t.Fatalf("result status %s, want %s", res.Status, StatusEnergyDrift)
	}
}

// The two drift statuses must be distinguishable strings for reporting.
func TestStatusStrings(t *testing.T) {
	if StatusOK.String() == StatusNormDrift.String() ||
		StatusOK.String() == StatusEnergyDrift.String() ||
		StatusNormDrift.String() == StatusEnergyDrift.String() {
		t.Fatalf("status strings must be distinct: %q %q %q",
			StatusOK, StatusNormDrift, StatusEnergyDrift)
	}
}
