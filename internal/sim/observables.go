package sim

import "math"

// Status classifies the trustworthiness of a run.
type Status int

const (
	// StatusOK means all conservation checks passed.
	StatusOK Status = iota
	// StatusNormDrift means the L2 norm drifted beyond tolerance.
	StatusNormDrift
	// StatusEnergyDrift means <H> drifted beyond tolerance.
	StatusEnergyDrift
)

func (s Status) String() string {
	switch s {
	case StatusOK:
		return "ok"
	case StatusNormDrift:
		return "norm_drift"
	case StatusEnergyDrift:
		return "energy_drift"
	default:
		return "unknown"
	}
}

// Diagnostics captures conservation quantities at one point in time.
type Diagnostics struct {
	Step        int
	Time        float64
	Norm        float64
	NormDrift   float64 // Norm - Norm at step 0
	Energy      float64
	EnergyDrift float64 // relative: (E - E0) / |E0|
	Status      Status
}

// Norm returns sum(|psi|^2) * cellArea. The reduction runs in a fixed
// serial order so the result is identical for any worker count.
func Norm(f *Field, cellArea float64) float64 {
	var acc float64
	for _, v := range f.Data {
		a := real(v)
		b := imag(v)
		acc += a*a + b*b
	}
	return acc * cellArea
}

// PotentialExpectation returns <psi|V|psi> (fixed serial reduction).
func PotentialExpectation(f *Field, V []float64, cellArea float64) float64 {
	var acc float64
	for i, v := range f.Data {
		a := real(v)
		b := imag(v)
		acc += V[i] * (a*a + b*b)
	}
	return acc * cellArea
}

// KineticExpectation returns <psi|T|psi> via spectral differentiation.
// Parseval: sum_x |psi|^2 = sum_k |psi_hat|^2 / (Nx*Ny).
func (s *Solver) KineticExpectation(f *Field) float64 {
	hat := f.Clone()
	FFT2D(hat, false, s.cfg.Workers)
	g := s.grid
	n := float64(g.Nx * g.Ny)
	coeff := s.cfg.Hbar * s.cfg.Hbar / (2 * s.cfg.Mass)
	var acc float64
	for i, v := range hat.Data {
		a := real(v)
		b := imag(v)
		acc += s.ksq[i] * (a*a + b*b)
	}
	return coeff * acc * g.CellArea() / n
}

// EnergyExpectation returns <psi|H|psi> = <T> + <V>.
func (s *Solver) EnergyExpectation(f *Field) float64 {
	return s.KineticExpectation(f) + PotentialExpectation(f, s.V, s.grid.CellArea())
}

// PositionExpectation returns (<x>, <y>) normalized by the total norm.
func PositionExpectation(f *Field, g *Grid) (x, y float64) {
	var norm, sx, sy float64
	for iy := 0; iy < g.Ny; iy++ {
		for ix := 0; ix < g.Nx; ix++ {
			v := f.Data[iy*g.Nx+ix]
			a := real(v)
			b := imag(v)
			p := a*a + b*b
			norm += p
			sx += g.X(ix) * p
			sy += g.Y(iy) * p
		}
	}
	if norm == 0 {
		return 0, 0
	}
	return sx / norm, sy / norm
}

// WidthX returns sqrt(<x^2> - <x>^2), the x standard deviation.
func WidthX(f *Field, g *Grid) float64 {
	var norm, sx, sxx float64
	for iy := 0; iy < g.Ny; iy++ {
		for ix := 0; ix < g.Nx; ix++ {
			v := f.Data[iy*g.Nx+ix]
			a := real(v)
			b := imag(v)
			p := a*a + b*b
			x := g.X(ix)
			norm += p
			sx += x * p
			sxx += x * x * p
		}
	}
	if norm == 0 {
		return 0
	}
	mean := sx / norm
	var2 := sxx/norm - mean*mean
	if var2 < 0 {
		var2 = 0
	}
	return math.Sqrt(var2)
}
