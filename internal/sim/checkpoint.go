package sim

import (
	"encoding/binary"
	"encoding/gob"
	"errors"
	"fmt"
	"hash/fnv"
	"io/fs"
	"math"
	"os"
	"path/filepath"
)

const (
	checkpointName = "checkpoint.gob"
	checkpointTmp  = "checkpoint.tmp"
)

// Checkpoint is a committed, resumable simulation state.
type Checkpoint struct {
	Step       int
	Time       float64
	Nx, Ny     int
	Data       []complex128
	Norm0      float64 // norm at step 0 (drift baseline)
	Energy0    float64 // <H> at step 0 (drift baseline)
	ConfigHash uint64
}

// HashConfig returns a stable hash of the state-evolution config fields:
// grid, physics, and time step. Fields that only affect scheduling, output
// cadence, or stopping criteria (Workers, OutputDir, Steps, SnapshotEvery,
// CheckpointEvery, tolerances) are excluded so they may change between a
// checkpoint and its resume.
func HashConfig(cfg Config) uint64 {
	h := fnv.New64a()
	var buf [8]byte
	putU := func(v uint64) {
		binary.LittleEndian.PutUint64(buf[:], v)
		h.Write(buf[:])
	}
	putF := func(v float64) { putU(math.Float64bits(v)) }
	putU(uint64(cfg.Nx))
	putU(uint64(cfg.Ny))
	putF(cfg.Lx)
	putF(cfg.Ly)
	putF(cfg.Mass)
	putF(cfg.Hbar)
	putF(cfg.Dt)
	return h.Sum64()
}

// SaveCheckpoint atomically commits ck into dir: the payload is written to
// a temporary file, fsynced, then renamed over the previous checkpoint.
// A crash can therefore leave either the old or the new checkpoint, never
// a partially written one.
func SaveCheckpoint(dir string, ck *Checkpoint) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp := filepath.Join(dir, checkpointTmp)
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	if err := gob.NewEncoder(f).Encode(ck); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmp, filepath.Join(dir, checkpointName)); err != nil {
		return err
	}
	syncDir(dir)
	return nil
}

// LoadCheckpoint returns the newest committed checkpoint in dir.
// It returns (nil, nil) when no committed checkpoint exists; leftover
// temporary files from interrupted writes are ignored and removed.
func LoadCheckpoint(dir string) (*Checkpoint, error) {
	// Clean up a partial write from a crashed SaveCheckpoint, if any.
	if err := os.Remove(filepath.Join(dir, checkpointTmp)); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return nil, err
	}
	f, err := os.Open(filepath.Join(dir, checkpointName))
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var ck Checkpoint
	if err := gob.NewDecoder(f).Decode(&ck); err != nil {
		return nil, fmt.Errorf("sim: decoding checkpoint: %w", err)
	}
	return &ck, nil
}

// syncDir best-effort fsyncs a directory so a rename is durable.
func syncDir(dir string) {
	d, err := os.Open(dir)
	if err != nil {
		return
	}
	d.Sync()
	d.Close()
}
