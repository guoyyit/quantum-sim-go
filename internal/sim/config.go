package sim

import "fmt"

// Default drift tolerances applied when the Config fields are not positive.
const (
	DefaultNormDriftTol   = 1e-9 // absolute drift of the L2 norm from 1
	DefaultEnergyDriftTol = 1e-4 // relative drift of <H> from its initial value
)

// Config describes a single simulation run.
type Config struct {
	Nx, Ny          int     // grid points per axis (must be powers of two)
	Lx, Ly          float64 // physical domain size
	Mass            float64 // particle mass
	Hbar            float64 // reduced Planck constant
	Dt              float64 // time step
	Steps           int     // number of time steps
	Workers         int     // parallel workers (<=0 means runtime.NumCPU())
	OutputDir       string  // directory for snapshots and checkpoints
	SnapshotEvery   int     // snapshot cadence in steps (0 disables snapshots)
	CheckpointEvery int     // checkpoint cadence in steps (0 disables checkpoints)
	NormDriftTol    float64 // max allowed |norm-1| before StatusNormDrift
	EnergyDriftTol  float64 // max allowed relative energy drift before StatusEnergyDrift
}

// Validate checks the configuration for consistency.
func (c Config) Validate() error {
	if !isPow2(c.Nx) || !isPow2(c.Ny) || c.Nx < 4 || c.Ny < 4 {
		return fmt.Errorf("sim: Nx=%d Ny=%d must be powers of two >= 4", c.Nx, c.Ny)
	}
	if c.Lx <= 0 || c.Ly <= 0 {
		return fmt.Errorf("sim: Lx=%g Ly=%g must be positive", c.Lx, c.Ly)
	}
	if c.Mass <= 0 {
		return fmt.Errorf("sim: Mass=%g must be positive", c.Mass)
	}
	if c.Hbar <= 0 {
		return fmt.Errorf("sim: Hbar=%g must be positive", c.Hbar)
	}
	if c.Dt <= 0 {
		return fmt.Errorf("sim: Dt=%g must be positive", c.Dt)
	}
	if c.Steps < 0 {
		return fmt.Errorf("sim: Steps=%d must be >= 0", c.Steps)
	}
	if c.SnapshotEvery < 0 || c.CheckpointEvery < 0 {
		return fmt.Errorf("sim: snapshot/checkpoint cadence must be >= 0")
	}
	if (c.SnapshotEvery > 0 || c.CheckpointEvery > 0) && c.OutputDir == "" {
		return fmt.Errorf("sim: OutputDir is required when snapshots or checkpoints are enabled")
	}
	return nil
}

func isPow2(n int) bool { return n > 0 && n&(n-1) == 0 }

// NormTol returns the effective norm drift tolerance.
func (c Config) NormTol() float64 {
	if c.NormDriftTol > 0 {
		return c.NormDriftTol
	}
	return DefaultNormDriftTol
}

// EnergyTol returns the effective relative energy drift tolerance.
func (c Config) EnergyTol() float64 {
	if c.EnergyDriftTol > 0 {
		return c.EnergyDriftTol
	}
	return DefaultEnergyDriftTol
}
