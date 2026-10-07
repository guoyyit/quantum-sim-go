package sim

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// Runs with different worker counts must produce bit-identical snapshots
// and identical diagnostics histories.
func TestParallelConsistency(t *testing.T) {
	workerCounts := []int{1, 2, 4, 8}
	type outcome struct {
		final   *Field
		history []Diagnostics
		snaps   map[string][]byte
	}
	run := func(workers int) outcome {
		dir := t.TempDir()
		cfg := Config{
			Nx: 64, Ny: 64, Lx: 20, Ly: 20,
			Mass: 1, Hbar: 1, Dt: 0.01, Steps: 60,
			Workers: workers, OutputDir: dir,
			SnapshotEvery: 10, CheckpointEvery: 25,
		}
		g := NewGrid(cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly)
		V := HarmonicPotential(g, 1, 1, 0, 0)
		psi0 := GaussianPacket(g, 2, -1, 1, 0.5, 0.25)
		res, err := Run(cfg, V, psi0)
		if err != nil {
			t.Fatalf("workers=%d: %v", workers, err)
		}
		if res.Status != StatusOK {
			t.Fatalf("workers=%d: status %s", workers, res.Status)
		}
		snaps := map[string][]byte{}
		matches, _ := filepath.Glob(filepath.Join(dir, "snapshot_*.gob"))
		for _, m := range matches {
			b, err := os.ReadFile(m)
			if err != nil {
				t.Fatal(err)
			}
			snaps[filepath.Base(m)] = b
		}
		return outcome{final: res.Final, history: res.History, snaps: snaps}
	}

	ref := run(workerCounts[0])
	for _, w := range workerCounts[1:] {
		got := run(w)
		if len(got.final.Data) != len(ref.final.Data) {
			t.Fatalf("workers=%d: final field size differs", w)
		}
		for i := range ref.final.Data {
			if got.final.Data[i] != ref.final.Data[i] {
				t.Fatalf("workers=%d: final field differs at %d: %v vs %v",
					w, i, got.final.Data[i], ref.final.Data[i])
			}
		}
		if len(got.history) != len(ref.history) {
			t.Fatalf("workers=%d: history length %d, want %d", w, len(got.history), len(ref.history))
		}
		for i := range ref.history {
			if got.history[i] != ref.history[i] {
				t.Fatalf("workers=%d: history[%d] differs: %+v vs %+v", w, i, got.history[i], ref.history[i])
			}
		}
		if len(got.snaps) != len(ref.snaps) {
			t.Fatalf("workers=%d: %d snapshots, want %d", w, len(got.snaps), len(ref.snaps))
		}
		for name, want := range ref.snaps {
			b, ok := got.snaps[name]
			if !ok {
				t.Fatalf("workers=%d: missing snapshot %s", w, name)
			}
			if !bytes.Equal(b, want) {
				t.Fatalf("workers=%d: snapshot %s bytes differ", w, name)
			}
		}
	}
}
