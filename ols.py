"""
# Multidimensional overlap-save method for fast-convolution

Features:

- 1D and 2D (both tested) and higher (untested) arrays
  - (Currently unsupported is convolving different-dimensional signals)
- Memory-mapped input/outputs fully supported (and tested)
- Relatively straightforward to paralellize each step of the algorithm
- Real-only, but readily modifiable to complex (convert `rfft` to `fft`)
- Equivalent to a `np.roll` of `scipy.signal.fftconvolve(mode='same')` (this is tested)
- Extensively unit-tested

The semantics of the convolution are as follows (the following is culled from
the unit test):
```py
import numpy as np

# Generate a 100x100 signal array and a 10x10 filter array
nx = 100
nh = 10
x = np.random.randint(-10, 10, size=(nx, nx)) + 1.0
h = np.random.randint(-10, 10, size=(nh, nh)) + 1.0

# Compute the linear convolution using the FFT, keep the first 100x100 samples
ngold = np.array(x.shape) + np.array(h.shape) - 1
expected = np.real(np.fft.ifft2(np.fft.fft2(x, ngold) *
                                np.conj(np.fft.fft2(h, ngold))))[:x.shape[0], :x.shape[1]]

# Use overlap-save, computing the output in 6x5-sized chunks. Instead of one
# huge FFT, we do a sequence of tiny ones
from ols import ols
actual = ols(x, h, [6, 5])

# The two will match
assert np.allclose(expected, actual)
```

Therefore, if you're computing fast-convolution as an IFFT of the product of FFTs, this module can
function as a drop-in replacement.

If you're using `scipy.signal.fftconvolve()` with `mode='same'`, then you have to roll the output
of this module to match what you have. You'll also have to throw away some data at the end due to
edge effects. Again culled from the unit-test:
```py
from scipy.signal import fftconvolve
conv = fftconvolve(x, h[::-1, ::-1], mode='same')
conv = np.roll(conv, [-(nh // 2)] * 2, [-1, -2])
assert np.allclose(conv[:-(nh // 2), :-(nh // 2)], gold[:-(nh // 2), :-(nh // 2)])
```
"""

import numpy as np
from nextprod import nextprod
from arrayRange import arrayRange
from typing import List


def prepareh(h, nfft: List[int]):
  """Pre-process a filter array
  
  Given a real filter array `h` and the length of the FFT `nfft`,
  returns the conjugated and FFT'd array. Needs to be computed
  only once before all steps of the overlap-save algorithm run.
  """
  return np.conj(np.fft.rfftn(h, nfft))


def olsStep(x, hfftconj, starts: List[int], lengths: List[int], nfft: List[int], nh: List[int]):
  """Implements a single step of the overlap-save algorithm

  Given an entire signal array `x` and the pre-transformed filter array
  `hfftconj` (i.e., the output of `prepareh`), compute a chunk of the total
  convolution. Specifically, the subarray of the total output starting at
  `starts`, with each dimension's length in `lengths`, is returned. The FFT
  length `nfft` (which was used in `prepareh`) is also required, as is `nh` the
  shape of the filter array (`h.shape`).

  For convenience, `lengths` is treated as a *maximum* length in each dimension,
  so `starts + lengths` is allowed to exceed the total size of `x`: the function
  won't read past the end of any arrays.

  The lists `starts`, `lengths`, `nft`, and `nh` are all required to be the same
  length, matching the number of dimensions of `x` and `hfftconj`.
  """
  assert len(x.shape) == len(hfftconj.shape)
  assert len(x.shape) == len(starts) and len(x.shape) == len(lengths)
  assert len(x.shape) == len(nfft) and len(x.shape) == len(nh)
  lengths = np.minimum(np.array(lengths), x.shape - np.array(starts))
  assert np.all(np.array(nfft) >= lengths + np.array(nh) - 1)
  slices = tuple(
      slice(start, start + length + nh - 1) for (start, length, nh) in zip(starts, lengths, nh))
  xpart = x[slices]
  output = np.fft.irfftn(np.fft.rfftn(xpart, nfft) * hfftconj, nfft)
  return output[tuple(slice(0, s) for s in lengths)]


def ols(x, h, size=None, nfft=None, out=None):
  assert len(x.shape) == len(h.shape)
  size = size or [4 * x for x in h.shape]
  nfft = nfft or [nextprod([2, 3, 5, 7], size + nh - 1) for size, nh in zip(size, h.shape)]
  assert len(x.shape) == len(size)
  assert len(x.shape) == len(nfft)

  hpre = prepareh(h, nfft)
  if out is None:
    out = np.zeros(x.shape, dtype=x.dtype)

  for tup in arrayRange([0 for _ in out.shape], out.shape, size):
    out[tup] = olsStep(x, hpre, [s.start for s in tup], size, nfft, h.shape)
  return out
