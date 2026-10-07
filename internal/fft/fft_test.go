package fft

import (
	"math"
	"math/cmplx"
	"testing"
)

func TestFFT1DRoundTrip(t *testing.T) {
	n := 64
	data := make([]complex128, n)
	for i := range data {
		data[i] = complex(math.Sin(float64(i)*0.7), math.Cos(float64(i)*1.3))
	}
	orig := append([]complex128(nil), data...)
	FFT1D(data, false)
	FFT1D(data, true)
	for i := range data {
		if d := cmplx.Abs(data[i] - orig[i]); d > 1e-12 {
			t.Fatalf("round trip mismatch at %d: got %v want %v (diff %g)", i, data[i], orig[i], d)
		}
	}
}

func TestFFT1DKnownValues(t *testing.T) {
	// DFT of a delta is a constant; DFT of a constant is a delta.
	n := 8
	delta := make([]complex128, n)
	delta[0] = 1
	FFT1D(delta, false)
	for i, v := range delta {
		if d := cmplx.Abs(v - 1); d > 1e-15 {
			t.Fatalf("delta transform[%d] = %v, want 1", i, v)
		}
	}
	constant := make([]complex128, n)
	for i := range constant {
		constant[i] = 1
	}
	FFT1D(constant, false)
	if d := cmplx.Abs(constant[0] - complex(float64(n), 0)); d > 1e-12 {
		t.Fatalf("constant transform[0] = %v, want %d", constant[0], n)
	}
	for i := 1; i < n; i++ {
		if d := cmplx.Abs(constant[i]); d > 1e-12 {
			t.Fatalf("constant transform[%d] = %v, want 0", i, constant[i])
		}
	}
}

func TestFFT2DParallelDeterministic(t *testing.T) {
	nx, ny := 32, 16
	base := make([]complex128, nx*ny)
	for i := range base {
		base[i] = complex(math.Sin(float64(i)*0.31), math.Cos(float64(i)*0.17))
	}
	serial := append([]complex128(nil), base...)
	FFT2D(serial, nx, ny, false, 1)
	for _, workers := range []int{2, 3, 4, 8} {
		par := append([]complex128(nil), base...)
		FFT2D(par, nx, ny, false, workers)
		for i := range par {
			if par[i] != serial[i] {
				t.Fatalf("workers=%d: element %d differs: %v != %v", workers, i, par[i], serial[i])
			}
		}
	}
}
