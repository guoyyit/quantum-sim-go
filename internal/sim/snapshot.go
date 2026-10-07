package sim

import (
	"encoding/gob"
	"fmt"
	"os"
	"path/filepath"
)

// Snapshot is one committed output record: the full complex wavefunction
// at one time slice. Encoding is deterministic, so snapshots from runs
// with identical inputs compare equal byte for byte.
type Snapshot struct {
	Step   int
	Time   float64
	Nx, Ny int
	Data   []complex128
}

// WriteSnapshot atomically writes s into dir and returns its path.
// The record is fsynced and renamed into place, so readers never observe
// a partially written snapshot.
func WriteSnapshot(dir string, s *Snapshot) (string, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	final := filepath.Join(dir, fmt.Sprintf("snapshot_%06d.gob", s.Step))
	tmp := final + ".tmp"
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	if err != nil {
		return "", err
	}
	if err := gob.NewEncoder(f).Encode(s); err != nil {
		f.Close()
		return "", err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return "", err
	}
	if err := f.Close(); err != nil {
		return "", err
	}
	if err := os.Rename(tmp, final); err != nil {
		return "", err
	}
	syncDir(dir)
	return final, nil
}

// ReadSnapshot loads a snapshot file.
func ReadSnapshot(path string) (*Snapshot, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var s Snapshot
	if err := gob.NewDecoder(f).Decode(&s); err != nil {
		return nil, fmt.Errorf("sim: decoding snapshot %s: %w", path, err)
	}
	return &s, nil
}
