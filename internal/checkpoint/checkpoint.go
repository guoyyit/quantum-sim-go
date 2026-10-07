// Package checkpoint provides atomic, resumable run state persistence.
//
// A checkpoint is committed by writing a temporary file, fsyncing it and
// atomically renaming it into place, so a crash can never leave a
// partially written checkpoint behind: readers only ever see complete
// committed files. On resume, the latest committed checkpoint is loaded
// and evolution continues from its step, so no time slice is repeated or
// skipped.
package checkpoint

import (
	"encoding/gob"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// State is a committed snapshot of an evolution run.
type State struct {
	Step int
	Time float64
	Nx   int
	Ny   int
	Data []complex128
}

// Manager writes checkpoints atomically into a directory.
type Manager struct {
	Dir   string
	Every int // commit period in steps
}

// NewManager creates a manager writing to dir every N steps.
func NewManager(dir string, every int) *Manager {
	if every <= 0 {
		every = 1
	}
	return &Manager{Dir: dir, Every: every}
}

// ShouldSave reports whether a checkpoint is due at the given step.
func (m *Manager) ShouldSave(step int) bool {
	return step%m.Every == 0
}

const (
	prefix = "checkpoint-"
	suffix = ".gob"
	tmpExt = ".tmp"
)

// Save atomically commits a checkpoint: the state is encoded to a
// temporary file in the same directory, fsynced, then renamed over any
// previous file with the same name.
func (m *Manager) Save(s State) error {
	if err := os.MkdirAll(m.Dir, 0o755); err != nil {
		return err
	}
	name := fmt.Sprintf("%s%08d%s", prefix, s.Step, suffix)
	final := filepath.Join(m.Dir, name)
	tmp := final + tmpExt
	f, err := os.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	if err := gob.NewEncoder(f).Encode(s); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, final); err != nil {
		os.Remove(tmp)
		return err
	}
	// Best effort: persist the directory entry of the rename.
	if d, err := os.Open(m.Dir); err == nil {
		d.Sync()
		d.Close()
	}
	return nil
}

// LoadLatest returns the most recent committed checkpoint, or (nil, nil)
// if none exists. Temporary (uncommitted) files are ignored and removed.
func (m *Manager) LoadLatest() (*State, error) {
	entries, err := os.ReadDir(m.Dir)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var steps []int
	for _, e := range entries {
		name := e.Name()
		if strings.HasSuffix(name, tmpExt) {
			// Uncommitted leftover from a crash: remove it.
			os.Remove(filepath.Join(m.Dir, name))
			continue
		}
		if !strings.HasPrefix(name, prefix) || !strings.HasSuffix(name, suffix) {
			continue
		}
		num := strings.TrimSuffix(strings.TrimPrefix(name, prefix), suffix)
		step, err := strconv.Atoi(num)
		if err != nil {
			continue
		}
		steps = append(steps, step)
	}
	if len(steps) == 0 {
		return nil, nil
	}
	sort.Ints(steps)
	latest := steps[len(steps)-1]
	return m.Load(latest)
}

// Load reads the committed checkpoint for the given step.
func (m *Manager) Load(step int) (*State, error) {
	name := fmt.Sprintf("%s%08d%s", prefix, step, suffix)
	f, err := os.Open(filepath.Join(m.Dir, name))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var s State
	if err := gob.NewDecoder(f).Decode(&s); err != nil {
		return nil, fmt.Errorf("checkpoint: corrupt file %s: %w", name, err)
	}
	return &s, nil
}
