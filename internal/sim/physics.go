package sim

import "math"

// ZeroPotential returns a vanishing potential on g.
func ZeroPotential(g *Grid) []float64 { return make([]float64, g.Nx*g.Ny) }

// HarmonicPotential returns V = (1/2) m omega^2 ((x-cx)^2 + (y-cy)^2).
func HarmonicPotential(g *Grid, mass, omega, cx, cy float64) []float64 {
	V := make([]float64, g.Nx*g.Ny)
	for iy := 0; iy < g.Ny; iy++ {
		for ix := 0; ix < g.Nx; ix++ {
			dx := g.X(ix) - cx
			dy := g.Y(iy) - cy
			V[iy*g.Nx+ix] = 0.5 * mass * omega * omega * (dx*dx + dy*dy)
		}
	}
	return V
}

// BarrierPotential returns a square barrier of the given height for
// x0 <= x <= x1 (independent of y).
func BarrierPotential(g *Grid, height, x0, x1 float64) []float64 {
	V := make([]float64, g.Nx*g.Ny)
	for iy := 0; iy < g.Ny; iy++ {
		for ix := 0; ix < g.Nx; ix++ {
			if x := g.X(ix); x >= x0 && x <= x1 {
				V[iy*g.Nx+ix] = height
			}
		}
	}
	return V
}

// GaussianPacket returns a normalized 2D Gaussian wave packet
//
//	psi = A exp(-r^2/(4 sigma^2)) exp(i (kx x + ky y)),  |psi|^2 has std sigma.
func GaussianPacket(g *Grid, x0, y0, sigma, kx, ky float64) *Field {
	f := NewField(g)
	norm := 1 / math.Sqrt(2*math.Pi*sigma*sigma)
	for iy := 0; iy < g.Ny; iy++ {
		for ix := 0; ix < g.Nx; ix++ {
			dx := g.X(ix) - x0
			dy := g.Y(iy) - y0
			env := norm * math.Exp(-(dx*dx+dy*dy)/(4*sigma*sigma))
			s, c := math.Sincos(kx*g.X(ix) + ky*g.Y(iy))
			f.Data[iy*g.Nx+ix] = complex(env*c, env*s)
		}
	}
	Normalize(f, g)
	return f
}

// PlaneWave returns the normalized plane wave exp(i(kx x + ky y))/sqrt(Lx Ly).
// kx, ky should be commensurate with the box (k = 2 pi n / L) to respect
// the periodic boundary conditions of the spectral method.
func PlaneWave(g *Grid, kx, ky float64) *Field {
	f := NewField(g)
	norm := 1 / math.Sqrt(g.Lx*g.Ly)
	for iy := 0; iy < g.Ny; iy++ {
		for ix := 0; ix < g.Nx; ix++ {
			s, c := math.Sincos(kx*g.X(ix) + ky*g.Y(iy))
			f.Data[iy*g.Nx+ix] = complex(norm*c, norm*s)
		}
	}
	return f
}

// AddFields returns a + b elementwise.
func AddFields(a, b *Field) *Field {
	out := a.Clone()
	for i := range out.Data {
		out.Data[i] += b.Data[i]
	}
	return out
}

// Scale returns c * f elementwise.
func Scale(f *Field, c complex128) *Field {
	out := f.Clone()
	for i := range out.Data {
		out.Data[i] *= c
	}
	return out
}

// Normalize scales f to unit L2 norm on g.
func Normalize(f *Field, g *Grid) { normalize(f, g) }
