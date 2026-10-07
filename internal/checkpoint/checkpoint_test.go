package checkpoint_test

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/guoyyit/quantum-sim-go/internal/checkpoint"
	"github.com/guoyyit/quantum-sim-go/internal/grid"
	"github.com/guoyyit/quantum-sim-go/internal/potential"
	"github.com/guoyyit/quantum-sim-go/internal/solver"
	"github.com/guoyyit/quantum-sim-go/internal/wavefunc"
)

func TestSaveLoadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	mgr := checkpoint.NewManager(dir, 5)
	st := checkpoint.State{Step: 42, Time: 0.42, Nx: 2, Ny: 3,
		Data: []complex128{1 + 2i, 3 + 4i, 5 + 6i, 7 + 8i, 9 + 10i, 11 + 12i}}
	if err := mgr.Save(st); err != nil {
		t.Fatal(err)
	}
	got, err := mgr.LoadLatest()
	if err != nil {
		t.Fatal(err)
	}
	if got == nil || got.Step != 42 || got.Time != 0.42 || got.Nx != 2 || got.Ny != 3 {
		t.Fatalf("loaded %+v, want step 42", got)
	}
	for i, v := range st.Data {
		if got.Data[i] != v {
			t.Fatalf("data[%d] = %v, want %v", i, got.Data[i], v)
		}
	}
}

func TestLoadLatestIgnoresPartialAndPicksHighest(t *testing.T) {
	dir := t.TempDir()
	mgr := checkpoint.NewManager(dir, 1)
	for _, step := range []int{10, 20, 30} {
		if err := mgr.Save(checkpoint.State{Step: step, Nx: 1, Ny: 1, Data: []complex128{complex(float64(step), 0)}}); err != nil {
			t.Fatal(err)
		}
	}
	// A partially written (uncommitted) checkpoint for a later step must
	// be ignored and cleaned up, never loaded.
	partial := filepath.Join(dir, "checkpoint-00000040.gob.tmp")
	if err := os.WriteFile(partial, []byte{0xde, 0xad}, 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := mgr.LoadLatest()
	if err != nil {
		t.Fatal(err)
	}
	if got == nil || got.Step != 30 {
		t.Fatalf("loaded step %v, want 30", got)
	}
	if _, err := os.Stat(partial); !os.IsNotExist(err) {
		t.Fatalf("partial file was not cleaned up")
	}
	// No checkpoints at all -> (nil, nil).
	empty := checkpoint.NewManager(t.TempDir(), 1)
	if st, err := empty.LoadLatest(); err != nil || st != nil {
		t.Fatalf("empty dir: got %v, %v", st, err)
	}
}

// runEvolve advances psi from startStep to totalSteps, committing
// checkpoints every ckEvery steps, and returns the final state plus the
// list of checkpoint steps actually committed.
func runEvolve(g *grid.Grid, pot *potential.Potential, psi *wavefunc.WaveFunction,
	startStep, totalSteps int, dt float64, mgr *checkpoint.Manager) []int {
	sv := solver.New(g, pot, solver.Params{Dt: dt, Steps: totalSteps - startStep, Workers: 4})
	var committed []int
	hooks := solver.Hooks{
		AfterStep: func(step int, tm float64, p *wavefunc.WaveFunction) {
			if mgr.ShouldSave(step) {
				if err := mgr.Save(checkpoint.State{
					Step: step, Time: tm, Nx: g.Nx, Ny: g.Ny,
					Data: append([]complex128(nil), p.Data...),
				}); err != nil {
					panic(err)
				}
				committed = append(committed, step)
			}
		},
	}
	if st := sv.Run(psi, startStep, hooks); st != solver.StatusOK {
		panic(st.String())
	}
	return committed
}

func TestResumeMatchesUninterrupted(t *testing.T) {
	const (
		nx, ny  = 32, 32
		L       = 24.0
		dt      = 0.01
		total   = 120
		ckEvery = 20
		crashAt = 74 // crash between committed checkpoints (60) and the end
	)
	g := grid.New(nx, ny, L, L)
	pot := potential.Harmonic(g, 1, 1)
	newPsi := func() *wavefunc.WaveFunction {
		w := wavefunc.FromFunc(g, func(x, y float64) complex128 {
			dx, dy := x-1.5, y+0.5
			return complex(math.Exp(-(dx*dx+dy*dy)/2), 0)
		})
		w.Normalize(g.Dx, g.Dy)
		return w
	}

	// Reference: uninterrupted run.
	refDir := t.TempDir()
	refPsi := newPsi()
	runEvolve(g, pot, refPsi, 0, total, dt, checkpoint.NewManager(refDir, ckEvery))

	// Crashed run: stops at crashAt, having committed up to step 60.
	crashDir := t.TempDir()
	mgr := checkpoint.NewManager(crashDir, ckEvery)
	crashPsi := newPsi()
	committed := runEvolve(g, pot, crashPsi, 0, crashAt, dt, mgr)
	if len(committed) == 0 || committed[len(committed)-1] != 60 {
		t.Fatalf("committed checkpoints = %v, want last = 60", committed)
	}

	// Resume: load the latest committed checkpoint and continue to total.
	st, err := mgr.LoadLatest()
	if err != nil || st == nil {
		t.Fatalf("LoadLatest: %v %v", st, err)
	}
	if st.Step != 60 {
		t.Fatalf("resuming from step %d, want 60", st.Step)
	}
	resPsi := &wavefunc.WaveFunction{Data: st.Data, Nx: st.Nx, Ny: st.Ny}
	runEvolve(g, pot, resPsi, st.Step, total, dt, mgr)

	// The resumed final state must be bitwise identical to the
	// uninterrupted run: no repeated or skipped time slices.
	for i := range refPsi.Data {
		if resPsi.Data[i] != refPsi.Data[i] {
			t.Fatalf("element %d differs after resume: %v != %v", i, resPsi.Data[i], refPsi.Data[i])
		}
	}
}
