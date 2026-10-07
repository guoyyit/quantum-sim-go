// Package config loads JSON run configurations.
package config

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config is the top level run configuration.
type Config struct {
	Grid struct {
		Nx, Ny int
		Lx, Ly float64
	} `json:"grid"`
	Evolution struct {
		Dt          float64 `json:"dt"`
		Steps       int     `json:"steps"`
		SampleEvery int     `json:"sample_every"`
		Workers     int     `json:"workers"`
		NormTol     float64 `json:"norm_tol"`
		EnergyTol   float64 `json:"energy_tol"`
	} `json:"evolution"`
	Potential struct {
		Kind          string  `json:"kind"` // "zero", "harmonic", "slit"
		OmegaX        float64 `json:"omega_x"`
		OmegaY        float64 `json:"omega_y"`
		BarrierHeight float64 `json:"barrier_height"`
		BarrierWidth  float64 `json:"barrier_width"`
		SlitSep       float64 `json:"slit_sep"`
		SlitWidth     float64 `json:"slit_width"`
	} `json:"potential"`
	Initial struct {
		Kind   string  `json:"kind"` // "gaussian"
		X0     float64 `json:"x0"`
		Y0     float64 `json:"y0"`
		SigmaX float64 `json:"sigma_x"`
		SigmaY float64 `json:"sigma_y"`
		Kx     float64 `json:"kx"`
		Ky     float64 `json:"ky"`
	} `json:"initial"`
	Checkpoint struct {
		Dir   string `json:"dir"`
		Every int    `json:"every"`
	} `json:"checkpoint"`
	Output struct {
		Dir string `json:"dir"`
	} `json:"output"`
}

// Load parses a JSON config file.
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var c Config
	if err := json.Unmarshal(data, &c); err != nil {
		return nil, fmt.Errorf("config: %w", err)
	}
	if c.Grid.Nx <= 0 || c.Grid.Ny <= 0 {
		return nil, fmt.Errorf("config: grid dimensions must be positive")
	}
	return &c, nil
}
