package sim

import (
	"context"
	"fmt"
)

// DriftError reports a conservation-law violation. It is returned together
// with the RunResult so callers can inspect the full history; the run stops
// at the first offending step instead of silently continuing.
type DriftError struct {
	Diag Diagnostics
}

func (e *DriftError) Error() string {
	return fmt.Sprintf("sim: %s at step %d (t=%g): norm drift=%g energy drift=%g",
		e.Diag.Status, e.Diag.Step, e.Diag.Time, e.Diag.NormDrift, e.Diag.EnergyDrift)
}

// RunResult summarizes a finished (or diagnosed) run.
type RunResult struct {
	Final   *Field
	History []Diagnostics
	Status  Status
}

// Run evolves psi0 under V, writing snapshots and checkpoints per cfg.
//
// Recovery: if cfg.OutputDir holds a committed checkpoint whose config hash
// matches, evolution resumes from checkpoint step+1. Because evolution is
// deterministic, any snapshot rewritten after a crash is byte-identical to
// the one written before it, so no time slice is ever duplicated or skipped.
func Run(cfg Config, V []float64, psi0 *Field) (*RunResult, error) {
	return RunContext(context.Background(), cfg, V, psi0)
}

// RunContext is Run with cancellation: when ctx is cancelled the run stops
// after the current step and returns ctx.Err(); every checkpoint committed
// so far remains valid for a later resume.
func RunContext(ctx context.Context, cfg Config, V []float64, psi0 *Field) (*RunResult, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if psi0 == nil || psi0.Nx != cfg.Nx || psi0.Ny != cfg.Ny {
		return nil, fmt.Errorf("sim: initial state does not match grid %dx%d", cfg.Nx, cfg.Ny)
	}
	solver, err := NewSolver(cfg, V)
	if err != nil {
		return nil, err
	}
	g := solver.Grid()

	res := &RunResult{Status: StatusOK}
	var psi *Field
	var norm0, e0 float64
	startStep := 0

	var ck *Checkpoint
	if cfg.OutputDir != "" {
		ck, err = LoadCheckpoint(cfg.OutputDir)
		if err != nil {
			return nil, err
		}
	}
	switch {
	case ck == nil:
		psi = psi0.Clone()
		norm0 = Norm(psi, g.CellArea())
		e0 = solver.EnergyExpectation(psi)
		res.History = append(res.History, Diagnostics{Step: 0, Norm: norm0, Energy: e0, Status: StatusOK})
		if cfg.SnapshotEvery > 0 {
			if _, err := WriteSnapshot(cfg.OutputDir, &Snapshot{Step: 0, Nx: g.Nx, Ny: g.Ny, Data: psi.Clone().Data}); err != nil {
				return nil, err
			}
		}
	case ck.ConfigHash != HashConfig(cfg):
		return nil, fmt.Errorf("sim: checkpoint config hash mismatch; refusing to resume")
	case ck.Nx != cfg.Nx || ck.Ny != cfg.Ny:
		return nil, fmt.Errorf("sim: checkpoint grid %dx%d does not match config %dx%d", ck.Nx, ck.Ny, cfg.Nx, cfg.Ny)
	default:
		psi = &Field{Nx: ck.Nx, Ny: ck.Ny, Data: ck.Data}
		norm0, e0 = ck.Norm0, ck.Energy0
		startStep = ck.Step
	}

	for step := startStep + 1; step <= cfg.Steps; step++ {
		if err := ctx.Err(); err != nil {
			res.Final = psi
			return res, err
		}
		solver.Step(psi)
		t := float64(step) * cfg.Dt

		if cfg.SnapshotEvery > 0 && step%cfg.SnapshotEvery == 0 {
			snap := &Snapshot{Step: step, Time: t, Nx: g.Nx, Ny: g.Ny, Data: psi.Clone().Data}
			if _, err := WriteSnapshot(cfg.OutputDir, snap); err != nil {
				return nil, err
			}
		}
		needDiag := step == cfg.Steps ||
			(cfg.SnapshotEvery > 0 && step%cfg.SnapshotEvery == 0) ||
			(cfg.CheckpointEvery > 0 && step%cfg.CheckpointEvery == 0)
		if needDiag {
			d := solver.diagnose(psi, step, t, norm0, e0, cfg)
			res.History = append(res.History, d)
			if d.Status != StatusOK {
				res.Final = psi
				res.Status = d.Status
				return res, &DriftError{Diag: d}
			}
		}
		if cfg.CheckpointEvery > 0 && step%cfg.CheckpointEvery == 0 {
			ck := &Checkpoint{
				Step: step, Time: t, Nx: g.Nx, Ny: g.Ny,
				Data: psi.Clone().Data, Norm0: norm0, Energy0: e0,
				ConfigHash: HashConfig(cfg),
			}
			if err := SaveCheckpoint(cfg.OutputDir, ck); err != nil {
				return nil, err
			}
		}
	}
	res.Final = psi
	return res, nil
}

// diagnose measures conservation quantities and classifies drift.
func (s *Solver) diagnose(psi *Field, step int, t, norm0, e0 float64, cfg Config) Diagnostics {
	n := Norm(psi, s.grid.CellArea())
	e := s.EnergyExpectation(psi)
	denom := abs(e0)
	if denom < 1e-300 {
		denom = 1
	}
	d := Diagnostics{
		Step:        step,
		Time:        t,
		Norm:        n,
		NormDrift:   n - norm0,
		Energy:      e,
		EnergyDrift: (e - e0) / denom,
		Status:      StatusOK,
	}
	if abs(d.NormDrift) > cfg.NormTol() {
		d.Status = StatusNormDrift
	} else if abs(d.EnergyDrift) > cfg.EnergyTol() {
		d.Status = StatusEnergyDrift
	}
	return d
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
