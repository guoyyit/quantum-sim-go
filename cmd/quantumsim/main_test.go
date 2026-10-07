package main_test

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

var binPath string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "quantumsim-bin")
	if err != nil {
		panic(err)
	}
	binPath = filepath.Join(dir, "quantumsim")
	build := exec.Command("go", "build", "-o", binPath, ".")
	build.Stderr = os.Stderr
	if err := build.Run(); err != nil {
		panic(err)
	}
	code := m.Run()
	os.RemoveAll(dir)
	os.Exit(code)
}

// writeConfig renders a JSON config for a harmonic-trap evolution.
func writeConfig(t *testing.T, dir string, steps int, ckDir, outDir string, workers int) string {
	t.Helper()
	cfg := fmt.Sprintf(`{
  "grid": {"Nx": 32, "Ny": 32, "Lx": 24, "Ly": 24},
  "evolution": {"dt": 0.01, "steps": %d, "sample_every": 20, "workers": %d},
  "potential": {"kind": "harmonic", "omega_x": 1, "omega_y": 1},
  "initial": {"kind": "gaussian", "x0": 1.5, "y0": -0.5, "sigma_x": 0.7, "sigma_y": 0.7, "kx": 0.3, "ky": 0},
  "checkpoint": {"dir": %q, "every": 20},
  "output": {"dir": %q}
}`, steps, workers, ckDir, outDir)
	path := filepath.Join(dir, "run.json")
	if err := os.WriteFile(path, []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func run(t *testing.T, args ...string) string {
	t.Helper()
	cmd := exec.Command(binPath, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%v failed: %v\n%s", cmd.Args, err, out)
	}
	return string(out)
}

func readFile(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func TestCLIEvolveParallelAndResume(t *testing.T) {
	root := t.TempDir()

	// Uninterrupted reference run, 1 worker.
	refOut := filepath.Join(root, "out-ref")
	refCfg := writeConfig(t, root, 120, filepath.Join(root, "ck-ref"), refOut, 1)
	run(t, "evolve", "-config", refCfg)

	// Parallel run with 8 workers: every frame must be byte identical.
	parOut := filepath.Join(root, "out-par")
	parCfg := writeConfig(t, root, 120, filepath.Join(root, "ck-par"), parOut, 8)
	run(t, "evolve", "-config", parCfg)
	for _, step := range []int{0, 20, 40, 60, 80, 100, 120} {
		name := fmt.Sprintf("frame-%08d.csv", step)
		a := readFile(t, filepath.Join(refOut, name))
		b := readFile(t, filepath.Join(parOut, name))
		if string(a) != string(b) {
			t.Fatalf("frame %s differs between workers=1 and workers=8", name)
		}
	}

	// Interrupted run: only 60 steps, leaving a committed checkpoint.
	ckDir := filepath.Join(root, "ck-crash")
	crashCfg := writeConfig(t, root, 60, ckDir, filepath.Join(root, "out-crash"), 1)
	run(t, "evolve", "-config", crashCfg)

	// Resume from the checkpoint and continue to step 120.
	resOut := filepath.Join(root, "out-resume")
	resCfg := writeConfig(t, root, 120, ckDir, resOut, 1)
	out := run(t, "evolve", "-config", resCfg, "-resume")
	if got := readFile(t, filepath.Join(resOut, "frame-00000120.csv")); string(got) != string(readFile(t, filepath.Join(refOut, "frame-00000120.csv"))) {
		t.Fatalf("resumed final frame differs from uninterrupted run")
	}
	if !strings.Contains(out, "resumed from checkpoint at step 60") {
		t.Fatalf("expected resume message, got:\n%s", out)
	}

	// Summary reports ok status.
	sum := readFile(t, filepath.Join(refOut, "summary.json"))
	if !strings.Contains(string(sum), `"status": "ok"`) {
		t.Fatalf("summary status not ok:\n%s", sum)
	}
}

func TestCLIGroundstate(t *testing.T) {
	root := t.TempDir()
	outDir := filepath.Join(root, "out-gs")
	cfg := fmt.Sprintf(`{
  "grid": {"Nx": 32, "Ny": 32, "Lx": 24, "Ly": 24},
  "potential": {"kind": "harmonic", "omega_x": 1, "omega_y": 1},
  "initial": {"kind": "gaussian", "x0": 0, "y0": 0, "sigma_x": 2, "sigma_y": 2},
  "output": {"dir": %q}
}`, outDir)
	path := filepath.Join(root, "gs.json")
	if err := os.WriteFile(path, []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	out := run(t, "groundstate", "-config", path, "-dtau", "0.01", "-tol", "1e-12")
	if !strings.Contains(out, "converged=true") {
		t.Fatalf("groundstate did not converge:\n%s", out)
	}
	// E0 = (wx + wy)/2 = 1 for hbar = m = 1.
	var energy float64
	if _, err := fmt.Sscanf(out, "energy=%g", &energy); err != nil {
		t.Fatalf("cannot parse energy from output:\n%s", out)
	}
	if d := energy - 1; d < -1e-4 || d > 1e-4 {
		t.Fatalf("ground state energy = %v, want ~1:\n%s", energy, out)
	}
	if _, err := os.Stat(filepath.Join(outDir, "residuals.csv")); err != nil {
		t.Fatalf("residual curve not written: %v", err)
	}
}
