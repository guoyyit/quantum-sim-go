// Package grid defines the 2D computational domain.
package grid

import "math"

// Grid describes a uniform 2D rectangular grid centered at the origin.
// The grid is periodic in both directions so that spectral (FFT based)
// operators can be applied.
type Grid struct {
	Nx, Ny int
	Lx, Ly float64
	Dx, Dy float64
	X, Y   []float64
	// Kx, Ky hold the angular wavenumbers in FFT ordering:
	// index n maps to 2*pi*n/L for n <= N/2 and 2*pi*(n-N)/L otherwise.
	Kx, Ky []float64
}

// New builds a grid with nx*ny points spanning lx*ly. nx and ny must be
// positive even numbers so that the FFT frequency grid is symmetric.
func New(nx, ny int, lx, ly float64) *Grid {
	if nx <= 0 || ny <= 0 {
		panic("grid: dimensions must be positive")
	}
	if nx%2 != 0 || ny%2 != 0 {
		panic("grid: dimensions must be even")
	}
	if lx <= 0 || ly <= 0 {
		panic("grid: lengths must be positive")
	}
	g := &Grid{Nx: nx, Ny: ny, Lx: lx, Ly: ly, Dx: lx / float64(nx), Dy: ly / float64(ny)}
	g.X = make([]float64, nx)
	for i := 0; i < nx; i++ {
		g.X[i] = (float64(i) - float64(nx)/2) * g.Dx
	}
	g.Y = make([]float64, ny)
	for j := 0; j < ny; j++ {
		g.Y[j] = (float64(j) - float64(ny)/2) * g.Dy
	}
	g.Kx = fftFreqs(nx, lx)
	g.Ky = fftFreqs(ny, ly)
	return g
}

// fftFreqs returns angular wavenumbers 2*pi*n/L in FFT ordering.
func fftFreqs(n int, l float64) []float64 {
	k := make([]float64, n)
	dk := 2 * math.Pi / l
	for i := 0; i < n; i++ {
		m := i
		if i > n/2 {
			m = i - n
		}
		k[i] = float64(m) * dk
	}
	return k
}
