package fft

import (
	"math"
	"sync"
)

// FFT1D computes the in-place discrete Fourier transform of data using an
// iterative radix-2 Cooley-Tukey algorithm. len(data) must be a power of
// two. inverse selects the sign of the exponent; the inverse transform is
// scaled by 1/n so that FFT1D(FFT1D(x, false), true) == x.
//
// The evaluation order is fixed, so results are bitwise reproducible.
func FFT1D(data []complex128, inverse bool) {
	n := len(data)
	if n == 0 || n&(n-1) != 0 {
		panic("fft: length must be a power of two")
	}
	// Bit-reversal permutation.
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j &^= bit
		}
		j |= bit
		if i < j {
			data[i], data[j] = data[j], data[i]
		}
	}
	for size := 2; size <= n; size <<= 1 {
		ang := 2 * math.Pi / float64(size)
		if inverse {
			ang = -ang
		}
		wlen := complex(math.Cos(ang), math.Sin(ang))
		half := size / 2
		for i := 0; i < n; i += size {
			w := complex(1, 0)
			for j := 0; j < half; j++ {
				u := data[i+j]
				v := data[i+j+half] * w
				data[i+j] = u + v
				data[i+j+half] = u - v
				w *= wlen
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

// FFT2D performs an in-place 2D FFT on row-major data of shape (nx, ny):
// first a 1D FFT along each row, then along each column. When workers > 1
// the independent rows (then columns) are distributed over a worker pool.
// Each transform is computed entirely by one goroutine in a fixed order,
// so the output is bitwise identical for any worker count.
func FFT2D(data []complex128, nx, ny int, inverse bool, workers int) {
	ParallelFor(nx, workers, ny, func(i int, _ []complex128) {
		FFT1D(data[i*ny:(i+1)*ny], inverse)
	})
	ParallelFor(ny, workers, nx, func(j int, scratch []complex128) {
		for i := 0; i < nx; i++ {
			scratch[i] = data[i*ny+j]
		}
		FFT1D(scratch, inverse)
		for i := 0; i < nx; i++ {
			data[i*ny+j] = scratch[i]
		}
	})
}

// ParallelFor runs f(i, scratch) for i in [0, n). Each worker owns a
// private scratch buffer of the given size. Every index is processed
// exactly once and f must not mutate shared state outside its own row or
// column, which keeps the combined result independent of the worker count.
func ParallelFor(n, workers, scratchSize int, f func(i int, scratch []complex128)) {
	if workers <= 1 || n <= 1 {
		scratch := make([]complex128, scratchSize)
		for i := 0; i < n; i++ {
			f(i, scratch)
		}
		return
	}
	if workers > n {
		workers = n
	}
	jobs := make(chan int)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			scratch := make([]complex128, scratchSize)
			for i := range jobs {
				f(i, scratch)
			}
		}()
	}
	for i := 0; i < n; i++ {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
}
