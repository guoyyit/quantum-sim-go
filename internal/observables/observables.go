// Package observables computes conserved quantities and expectations.
package observables

import (
	"math"

	"github.com/guoyyit/quantum-sim-go/internal/fft"
	"github.com/guoyyit/quantum-sim-go/internal/grid"
	"github.com/guoyyit/quantum-sim-go/internal/potential"
	"github.com/guoyyit/quantum-sim-go/internal/wavefunc"
)

// Norm returns the L2 norm of w on the grid.
func Norm(w *wavefunc.WaveFunction, g *grid.Grid) float64 {
	return math.Sqrt(w.Norm2(g.Dx, g.Dy))
}

// Kinetic returns the kinetic energy expectation <psi| -hbar^2/(2m) laplacian |psi>
// evaluated spectrally: sum_k (hbar k)^2/(2m) |psi~(k)|^2 with the Parseval
// weight. The input wavefunction is not modified.
func Kinetic(w *wavefunc.WaveFunction, g *grid.Grid, mass, hbar float64) float64 {
	n := g.Nx * g.Ny
	buf := make([]complex128, n)
	copy(buf, w.Data)
	fft.FFT2D(buf, g.Nx, g.Ny, false, 1)
	// Parseval: sum_x |f(x)|^2 = (1/N) sum_k |F(k)|^2 for the unscaled
	// forward transform. Physical cell weight dx*dy then applies.
	var s float64
	for i := 0; i < g.Nx; i++ {
		kx := g.Kx[i]
		for j := 0; j < g.Ny; j++ {
			ky := g.Ky[j]
			k2 := kx*kx + ky*ky
			v := buf[i*g.Ny+j]
			s += k2 * (real(v)*real(v) + imag(v)*imag(v))
		}
	}
	return hbar * hbar / (2 * mass) * s * g.Dx * g.Dy / float64(n)
}

// PotentialEnergy returns <psi| V |psi>.
func PotentialEnergy(w *wavefunc.WaveFunction, p *potential.Potential, g *grid.Grid) float64 {
	var s float64
	for i, v := range w.Data {
		s += p.V[i] * (real(v)*real(v) + imag(v)*imag(v))
	}
	return s * g.Dx * g.Dy
}

// Energy returns the total energy expectation <H> = <T> + <V>.
func Energy(w *wavefunc.WaveFunction, g *grid.Grid, p *potential.Potential, mass, hbar float64) float64 {
	return Kinetic(w, g, mass, hbar) + PotentialEnergy(w, p, g)
}
