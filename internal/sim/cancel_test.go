package sim

import (
	"context"
	"errors"
	"testing"
)

// A cancelled run must stop promptly, leave the last committed checkpoint
// intact, and a later run must resume from it to the correct final state.
func TestRunContextCancellation(t *testing.T) {
	g := NewGrid(32, 32, 20, 20)
	V := HarmonicPotential(g, 1, 1, 0, 0)
	psi0 := GaussianPacket(g, 1, 0, 1, 0.2, 0)

	// Reference: uninterrupted 100-step run.
	dirRef := t.TempDir()
	ref, err := Run(recoveryCfg(dirRef, 100), V, psi0)
	if err != nil {
		t.Fatal(err)
	}

	// First segment commits a checkpoint at step 40.
	dir := t.TempDir()
	if _, err := Run(recoveryCfg(dir, 40), V, psi0); err != nil {
		t.Fatal(err)
	}
	// Cancelled before it starts: must return ctx.Err() and commit nothing.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = RunContext(ctx, recoveryCfg(dir, 100), V, psi0)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	ck, err := LoadCheckpoint(dir)
	if err != nil {
		t.Fatal(err)
	}
	if ck == nil || ck.Step != 40 {
		t.Fatalf("cancelled run corrupted checkpoint: %+v", ck)
	}
	// Resume completes and matches the reference bit for bit.
	res, err := Run(recoveryCfg(dir, 100), V, psi0)
	if err != nil {
		t.Fatal(err)
	}
	for i := range ref.Final.Data {
		if res.Final.Data[i] != ref.Final.Data[i] {
			t.Fatalf("final state after cancel+resume differs at %d", i)
		}
	}
}
