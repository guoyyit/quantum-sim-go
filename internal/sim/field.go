package sim

import "math"

// Grid describes the 2D spatial grid and its Fourier-space dual.
// The domain is periodic: x in [-Lx/2, Lx/2), y in [-Ly/2, Ly/2).
type Grid struct {
	Nx, Ny int
	Lx, Ly float64
	Dx, Dy float64
	Kx, Ky []float64 // angular wavenumbers, ordered to match the FFT layout
}

// NewGrid builds a Grid including the angular wavenumber axes.
func NewGrid(nx, ny int, lx, ly float64) *Grid {
	g := &Grid{
		Nx: nx, Ny: ny,
		Lx: lx, Ly: ly,
		Dx: lx / float64(nx),
		Dy: ly / float64(ny),
		Kx: fftFreqs(nx, lx),
		Ky: fftFreqs(ny, ly),
	}
	return g
}

// fftFreqs returns angular wavenumbers 2*pi*k/L in FFT order:
// indices [0..n/2-1] hold k=0..n/2-1, indices [n/2..n-1] hold k=-n/2..-1.
func fftFreqs(n int, l float64) []float64 {
	k := make([]float64, n)
	for i := 0; i < n; i++ {
		ki := i
		if i > n/2 {
			ki = i - n
		}
		k[i] = 2 * math.Pi * float64(ki) / l
	}
	return k
}

// X returns the physical x coordinate of column ix.
func (g *Grid) X(ix int) float64 { return (float64(ix) - float64(g.Nx)/2) * g.Dx }

// Y returns the physical y coordinate of row iy.
func (g *Grid) Y(iy int) float64 { return (float64(iy) - float64(g.Ny)/2) * g.Dy }

// CellArea returns the area of one grid cell.
func (g *Grid) CellArea() float64 { return g.Dx * g.Dy }

// Field is a complex-valued wavefunction sampled on a Grid.
// Data is row-major: element (ix, iy) lives at Data[iy*Nx+ix].
type Field struct {
	Nx, Ny int
	Data   []complex128
}

// NewField allocates a zeroed Field on grid g.
func NewField(g *Grid) *Field {
	return &Field{Nx: g.Nx, Ny: g.Ny, Data: make([]complex128, g.Nx*g.Ny)}
}

// At returns the value at grid point (ix, iy).
func (f *Field) At(ix, iy int) complex128 { return f.Data[iy*f.Nx+ix] }

// Set assigns v to grid point (ix, iy).
func (f *Field) Set(ix, iy int, v complex128) { f.Data[iy*f.Nx+ix] = v }

// Clone returns a deep copy of f.
func (f *Field) Clone() *Field {
	out := &Field{Nx: f.Nx, Ny: f.Ny, Data: make([]complex128, len(f.Data))}
	copy(out.Data, f.Data)
	return out
}
