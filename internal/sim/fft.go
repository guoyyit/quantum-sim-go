package sim

import (
	"math"
	"runtime"
	"sync"
)

// fft1D performs an in-place radix-2 Cooley-Tukey FFT.
// Forward:  X[k] = sum_j x[j] exp(-2*pi*i*j*k/n)
// Inverse:  x[j] = (1/n) sum_k X[k] exp(+2*pi*i*j*k/n)
// The evaluation order is fixed, so results are bit-reproducible.
func fft1D(data []complex128, inverse bool) {
	n := len(data)
	if n <= 1 {
		return
	}
	// Bit-reversal permutation.
	j := 0
	for i := 0; i < n-1; i++ {
		if i < j {
			data[i], data[j] = data[j], data[i]
		}
		k := n >> 1
		for k <= j {
			j -= k
			k >>= 1
		}
		j += k
	}
	// Butterfly stages.
	sign := -1.0
	if inverse {
		sign = 1.0
	}
	for size := 2; size <= n; size <<= 1 {
		half := size >> 1
		for start := 0; start < n; start += size {
			for m := 0; m < half; m++ {
				s, c := math.Sincos(sign * 2 * math.Pi * float64(m) / float64(size))
				w := complex(c, s)
				a := data[start+m]
				b := data[start+m+half] * w
				data[start+m] = a + b
				data[start+m+half] = a - b
			}
		}
	}
	if inverse {
		inv := complex(1/float64(n), 0)
		for i := range data {
			data[i] *= inv
		}
	}
}

// parallelFor runs fn(i) for i in [0, n) using at most workers goroutines.
// Each i is handled independently, so the result never depends on workers.
func parallelFor(n, workers int, fn func(i int)) {
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	if workers < 1 {
		workers = 1
	}
	if workers > n {
		workers = n
	}
	if workers == 1 {
		for i := 0; i < n; i++ {
			fn(i)
		}
		return
	}
	var wg sync.WaitGroup
	chunk := (n + workers - 1) / workers
	for w := 0; w < workers; w++ {
		lo := w * chunk
		hi := lo + chunk
		if hi > n {
			hi = n
		}
		if lo >= hi {
			break
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for i := lo; i < hi; i++ {
				fn(i)
			}
		}(lo, hi)
	}
	wg.Wait()
}

// FFT2D transforms f in place as separable 1D FFTs over rows then columns.
// Work is partitioned over independent rows/columns, so the output is
// bit-identical for any worker count.
func FFT2D(f *Field, inverse bool, workers int) {
	nx, ny := f.Nx, f.Ny
	parallelFor(ny, workers, func(iy int) {
		fft1D(f.Data[iy*nx:(iy+1)*nx], inverse)
	})
	parallelFor(nx, workers, func(ix int) {
		col := make([]complex128, ny)
		for iy := 0; iy < ny; iy++ {
			col[iy] = f.Data[iy*nx+ix]
		}
		fft1D(col, inverse)
		for iy := 0; iy < ny; iy++ {
			f.Data[iy*nx+ix] = col[iy]
		}
	})
}
