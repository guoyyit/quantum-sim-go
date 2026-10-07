package sim

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func recoveryCfg(dir string, steps int) Config {
	return Config{
		Nx: 32, Ny: 32, Lx: 20, Ly: 20,
		Mass: 1, Hbar: 1, Dt: 0.01, Steps: steps,
		Workers: 2, OutputDir: dir,
		SnapshotEvery: 10, CheckpointEvery: 20,
	}
}

func readSnapshots(t *testing.T, dir string) map[string][]byte {
	t.Helper()
	out := map[string][]byte{}
	matches, err := filepath.Glob(filepath.Join(dir, "snapshot_*.gob"))
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range matches {
		b, err := os.ReadFile(m)
		if err != nil {
			t.Fatal(err)
		}
		out[filepath.Base(m)] = b
	}
	return out
}

// A run interrupted after its last committed checkpoint must resume from
// exactly that checkpoint and produce bit-identical output to an
// uninterrupted run: no duplicated or skipped time slices.
func TestCheckpointResumeMatchesUninterrupted(t *testing.T) {
	g := NewGrid(32, 32, 20, 20)
	V := HarmonicPotential(g, 1, 1, 0, 0)
	psi0 := GaussianPacket(g, 1.5, 0.5, 1, 0.3, 0)

	// Uninterrupted reference.
	dirA := t.TempDir()
	resA, err := Run(recoveryCfg(dirA, 100), V, psi0)
	if err != nil {
		t.Fatal(err)
	}
	snapsA := readSnapshots(t, dirA)

	// "Crashed" run: stops at step 55, last committed checkpoint is step 40.
	dirB := t.TempDir()
	if _, err := Run(recoveryCfg(dirB, 55), V, psi0); err != nil {
		t.Fatal(err)
	}
	ck, err := LoadCheckpoint(dirB)
	if err != nil {
		t.Fatal(err)
	}
	if ck == nil || ck.Step != 40 {
		t.Fatalf("expected committed checkpoint at step 40, got %+v", ck)
	}

	// Resume to step 100 from the checkpoint.
	resB, err := Run(recoveryCfg(dirB, 100), V, psi0)
	if err != nil {
		t.Fatal(err)
	}
	if resB.Status != StatusOK {
		t.Fatalf("resumed run status %s", resB.Status)
	}

	// Final states must be bit-identical.
	for i := range resA.Final.Data {
		if resA.Final.Data[i] != resB.Final.Data[i] {
			t.Fatalf("final state differs at %d: %v vs %v", i, resA.Final.Data[i], resB.Final.Data[i])
		}
	}
	// Every snapshot of the uninterrupted run must exist, byte-identical.
	snapsB := readSnapshots(t, dirB)
	if len(snapsB) != len(snapsA) {
		t.Fatalf("snapshot count %d, want %d", len(snapsB), len(snapsA))
	}
	for name, want := range snapsA {
		got, ok := snapsB[name]
		if !ok {
			t.Fatalf("missing snapshot %s after resume", name)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("snapshot %s differs after resume", name)
		}
	}
	// The resumed run must not re-diagnose committed steps.
	for _, d := range resB.History {
		if d.Step <= 40 {
			t.Fatalf("resumed run repeated time slice %d", d.Step)
		}
	}
}

// A crashed checkpoint write leaves a .tmp file; it must be ignored and
// cleaned up, never mistaken for a committed checkpoint.
func TestPartialCheckpointWriteIgnored(t *testing.T) {
	dir := t.TempDir()
	g := NewGrid(32, 32, 20, 20)
	V := HarmonicPotential(g, 1, 1, 0, 0)
	psi0 := GaussianPacket(g, 1, 0, 1, 0, 0)
	if _, err := Run(recoveryCfg(dir, 40), V, psi0); err != nil {
		t.Fatal(err)
	}
	// Simulate a crashed write: garbage in the temp file.
	tmp := filepath.Join(dir, "checkpoint.tmp")
	if err := os.WriteFile(tmp, []byte("partial garbage"), 0o644); err != nil {
		t.Fatal(err)
	}
	ck, err := LoadCheckpoint(dir)
	if err != nil {
		t.Fatal(err)
	}
	if ck == nil || ck.Step != 40 {
		t.Fatalf("expected committed checkpoint at step 40, got %+v", ck)
	}
	if _, err := os.Stat(tmp); !os.IsNotExist(err) {
		t.Fatalf("stale temp file was not cleaned up")
	}
}

// Resuming with different physics must be refused, not silently continued.
func TestCheckpointConfigMismatch(t *testing.T) {
	dir := t.TempDir()
	g := NewGrid(32, 32, 20, 20)
	V := HarmonicPotential(g, 1, 1, 0, 0)
	psi0 := GaussianPacket(g, 1, 0, 1, 0, 0)
	if _, err := Run(recoveryCfg(dir, 40), V, psi0); err != nil {
		t.Fatal(err)
	}
	bad := recoveryCfg(dir, 100)
	bad.Dt = 0.02
	if _, err := Run(bad, V, psi0); err == nil || !strings.Contains(err.Error(), "hash mismatch") {
		t.Fatalf("expected hash mismatch error, got %v", err)
	}
}

// A corrupted committed checkpoint must surface as an error.
func TestCorruptCheckpointDetected(t *testing.T) {
	dir := t.TempDir()
	g := NewGrid(32, 32, 20, 20)
	V := HarmonicPotential(g, 1, 1, 0, 0)
	psi0 := GaussianPacket(g, 1, 0, 1, 0, 0)
	if _, err := Run(recoveryCfg(dir, 40), V, psi0); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "checkpoint.gob"), []byte("corrupt"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCheckpoint(dir); err == nil {
		t.Fatalf("corrupt checkpoint was not detected")
	}
}
