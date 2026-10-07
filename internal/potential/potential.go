// Package potential builds external potential energy landscapes.
package potential

import (
	"math"

	"github.com/guoyyit/quantum-sim-go/internal/grid"
)

// Potential stores real potential values in row-major order, matching
// the wavefunction layout.
type Potential struct {
	V  []float64
	Nx int
	Ny int
}

// Zero returns a vanishing potential.
func Zero(g *grid.Grid) *Potential {
	return &Potential{V: make([]float64, g.Nx*g.Ny), Nx: g.Nx, Ny: g.Ny}
}

// FromFunc samples f on the grid.
func FromFunc(g *grid.Grid, f func(x, y float64) float64) *Potential {
	p := &Potential{V: make([]float64, g.Nx*g.Ny), Nx: g.Nx, Ny: g.Ny}
	for i := 0; i < g.Nx; i++ {
		for j := 0; j < g.Ny; j++ {
			p.V[i*g.Ny+j] = f(g.X[i], g.Y[j])
		}
	}
	return p
}

// Harmonic returns V(x,y) = 0.5*(omegaX^2 x^2 + omegaY^2 y^2) for unit mass.
func Harmonic(g *grid.Grid, omegaX, omegaY float64) *Potential {
	return FromFunc(g, func(x, y float64) float64 {
		return 0.5 * (omegaX*omegaX*x*x + omegaY*omegaY*y*y)
	})
}

// SlitBarrier returns a wall of the given height for |x| < width/2 with two
// Gaussian-tapered slits centered at y = +/- slitSep/2 of width slitWidth.
// It is a convenient potential for double-slit interference experiments.
func SlitBarrier(g *grid.Grid, height, width, slitSep, slitWidth float64) *Potential {
	return FromFunc(g, func(x, y float64) float64 {
		if x < -width/2 || x > width/2 {
			return 0
		}
		if slitWidth <= 0 {
			return height
		}
		// Smooth slits: attenuate the wall by (1 - exp(-d^2/(2 s^2))) where d
		// is the distance to the nearest slit center.
		d1 := y - slitSep/2
		d2 := y + slitSep/2
		d := d1
		if math.Abs(d2) < math.Abs(d1) {
			d = d2
		}
		block := 1 - math.Exp(-d*d/(2*slitWidth*slitWidth))
		return height * block
	})
}

// At returns the potential at grid indices (i, j).
func (p *Potential) At(i, j int) float64 { return p.V[i*p.Ny+j] }
