// Package wavefunc holds the complex-valued wavefunction on a grid.
package wavefunc

import (
	"math"

	"github.com/guoyyit/quantum-sim-go/internal/grid"
)

// WaveFunction stores complex amplitudes in row-major order:
// index (i, j) maps to Data[i*Ny+j] where i is the x index and j the y index.
type WaveFunction struct {
	Data []complex128
	Nx   int
	Ny   int
}

// New allocates a zeroed wavefunction.
func New(nx, ny int) *WaveFunction {
	return &WaveFunction{Data: make([]complex128, nx*ny), Nx: nx, Ny: ny}
}

// FromFunc samples f on the grid.
func FromFunc(g *grid.Grid, f func(x, y float64) complex128) *WaveFunction {
	w := New(g.Nx, g.Ny)
	for i := 0; i < g.Nx; i++ {
		for j := 0; j < g.Ny; j++ {
			w.Data[i*g.Ny+j] = f(g.X[i], g.Y[j])
		}
	}
	return w
}

// At returns the amplitude at grid indices (i, j).
func (w *WaveFunction) At(i, j int) complex128 { return w.Data[i*w.Ny+j] }

// Set assigns the amplitude at grid indices (i, j).
func (w *WaveFunction) Set(i, j int, v complex128) { w.Data[i*w.Ny+j] = v }

// Clone returns a deep copy.
func (w *WaveFunction) Clone() *WaveFunction {
	c := &WaveFunction{Data: make([]complex128, len(w.Data)), Nx: w.Nx, Ny: w.Ny}
	copy(c.Data, w.Data)
	return c
}

// Norm2 returns the squared L2 norm with cell weights dx*dy.
func (w *WaveFunction) Norm2(dx, dy float64) float64 {
	var s float64
	for _, v := range w.Data {
		s += real(v)*real(v) + imag(v)*imag(v)
	}
	return s * dx * dy
}

// Normalize rescales to unit norm. A zero wavefunction is left unchanged.
func (w *WaveFunction) Normalize(dx, dy float64) {
	n2 := w.Norm2(dx, dy)
	if n2 == 0 {
		return
	}
	inv := 1 / math.Sqrt(n2)
	for i := range w.Data {
		w.Data[i] *= complex(inv, 0)
	}
}

// Inner returns the weighted inner product <a|b> = sum conj(a_i) b_i dx dy.
func Inner(a, b *WaveFunction, dx, dy float64) complex128 {
	var re, im float64
	for i, av := range a.Data {
		bv := b.Data[i]
		// conj(a)*b = (ar - i ai)(br + i bi)
		re += real(av)*real(bv) + imag(av)*imag(bv)
		im += real(av)*imag(bv) - imag(av)*real(bv)
	}
	return complex(re, im) * complex(dx*dy, 0)
}
