package sim

import (
	"math"
	"testing"
)

// deterministic pseudo-random field (no external seed needed).
func testField(nx, ny int) *Field {
	f := &Field{Nx: nx, Ny: ny, Data: make([]complex128, nx*ny)}
	for i := range f.Data {
		x := float64(i) * 0.6180339887498949
		f.Data[i] = complex(math.Sin(x*7.1), math.Cos(x*3.7))
	}
	return f
}

func TestFFT1DRoundTrip(t *testing.T) {
	data := make([]complex128, 64)
	for i := range data {
		data[i] = complex(math.Sin(float64(i)), math.Cos(float64(i)*0.5))
	}
	orig := append([]complex128(nil), data...)
	fft1D(data, false)
	fft1D(data, true)
	for i := range data {
		if d := data[i] - orig[i]; abs(real(d)) > 1e-12 || abs(imag(d)) > 1e-12 {
			t.Fatalf("round trip mismatch at %d: got %v want %v", i, data[i], orig[i])
		}
	}
}

func TestFFT2DMatchesNaiveDFT(t *testing.T) {
	const n = 8
	f := testField(n, n)
	got := f.Clone()
	FFT2D(got, false, 1)
	// Naive 2D DFT at a few selected (kx, ky).
	for _, k := range [][2]int{{0, 0}, {1, 0}, {0, 1}, {3, 5}, {7, 7}} {
		var want complex128
		for iy := 0; iy < n; iy++ {
			for ix := 0; ix < n; ix++ {
				ang := -2 * math.Pi * (float64(k[0]*ix) + float64(k[1]*iy)) / n
				s, c := math.Sincos(ang)
				want += f.Data[iy*n+ix] * complex(c, s)
			}
		}
		if d := got.Data[k[1]*n+k[0]] - want; abs(real(d)) > 1e-9 || abs(imag(d)) > 1e-9 {
			t.Errorf("FFT mismatch at k=%v: got %v want %v", k, got.Data[k[1]*n+k[0]], want)
		}
	}
}

func TestFFT2DParallelBitIdentical(t *testing.T) {
	ref := testField(32, 32)
	FFT2D(ref, false, 1)
	for _, w := range []int{2, 3, 4, 8, 16} {
		got := testField(32, 32)
		FFT2D(got, false, w)
		for i := range got.Data {
			if got.Data[i] != ref.Data[i] {
				t.Fatalf("workers=%d: element %d differs: %v vs %v", w, i, got.Data[i], ref.Data[i])
			}
		}
	}
}
