# -*- coding: utf-8 -*-

import scipy.fft as sf
import numpy as np
from nextprod import nextprod
from ols import ols
from typing import List, Optional


def texshadeFFT(x: np.ndarray, alpha: float) -> np.ndarray:
  """FFT-based texture shading elevation

  Given an array `x` of elevation data and an `alpha` > 0, apply the
  texture-shading algorithm using the full (real-only) FFT: the entire `x` array
  will be FFT'd.

  `alpha` is the shading detail factor, i.e., the power of the
  fractional-Laplacian operator. `alpha=0` means no detail (output is the
  input). `alpha=2.0` is the full (non-fractional) Laplacian operator and is
  probably too high. `alpha <= 1.0` seem aesthetically pleasing.

  Returns an array the same dimensions as `x` that contains the texture-shaded
  version of the input array.

  If `x` is memory-mapped and/or your system doesn't have 5x `x`'s memory
  available, consider using `texshade.texshadeSpatial`, which implements a
  low-memory version of the algorithm by approximating the frequency response of
  the fractional-Laplacian filter with a finite impulse response filter applied
  in the spatial-domain.

  Implementation note: this function uses Scipy's FFTPACK routines (in
  `scipy.fftpack`) instead of Numpy's FFT (`numpy.fft`) because the former can
  return single-precision float32. In newer versions of Numpy/Scipy, this
  advantage may have evaporated [1], [2].

  [1] https://github.com/numpy/numpy/issues/6012
  [2] https://github.com/scipy/scipy/issues/2487
  """
  Nyx = [nextprod([2, 3, 5, 7], x) for x in x.shape]

  # Generate filter in the frequency domain
  fy = sf.fftfreq(Nyx[0])[:, np.newaxis].astype(x.dtype)
  fx = sf.rfftfreq(Nyx[1])[np.newaxis, :].astype(x.dtype)
  H2 = (fx**2 + fy**2)**(alpha / 2.0)

  # Compute the FFT of the input and apply the filter
  xr = sf.rfft2(x, s=Nyx) * H2
  H2 = None  # potentially trigger GC here to reclaim H2's memory
  xr = sf.irfft2(xr)
  # Return the same size as input
  return xr[:x.shape[0], :x.shape[1]]


def makeFilter(shape: List[int], alpha: float, dtype=float) -> np.ndarray:
  assert 1 <= len(shape) <= 2, "shape must be one or two elements"
  if len(shape) == 1:
    shape = [shape[0], shape[0]]

  # Generate filter in the frequency domain
  fy = sf.fftfreq(shape[0])[:, np.newaxis].astype(dtype)
  fx = sf.rfftfreq(shape[1])[np.newaxis, :].astype(dtype)
  H2 = (fx**2 + fy**2)**(alpha / 2.0)

  return sf.ifftshift(sf.irfft2(H2))


def texshadeSpatial(
    x: np.ndarray,
    alpha: Optional[float] = None,
    nDiameter: Optional[int] = None,
    filter: Optional[np.ndarray] = None,
    # ols kwargs
    size=None,
    nfft=None,
    out=None,
) -> np.ndarray:
  """Low-memory approximation of the texture shading algorithm

  Unlike `texshade.texshadeFFT`, which computes an FFT of the entire input
  elevation array `x` and applies the fractional-Laplacian filter in the
  frequency domain, this function approximates that frequency response with a
  spatial-domain finite impulse response (FIR) filter that is applied in the
  spatial domain via fast-convolution (overlap-save method). This allows `x` to
  be memory-mapped and/or very large relative to the amount of free system
  memory.

  `alpha` is the shading detail factor, i.e., the power of the
  fractional-Laplacian operator. `alpha=0` means no detail (output is the
  input). `alpha=2.0` is the full (non-fractional) Laplacian operator and is
  probably too high. `alpha <= 1.0` seem aesthetically pleasing.

  Returns an array the same dimensions as `x` that contains the texture-shaded
  version of the input array.

  `nDiameterOrFilter`
  
  **Overlap-save keyword args** passed to `ols.ols` (this function is in the
  `overlap-save` module on PyPI):

  - `size`
  - `nfft`
  - `out`

  `size`, a 2-list, specifies the size of the sub-arrays of the texture-shaded
  output to compute in each overlap-save step, while `nfft` (also a 2-list) is
  the size of the zero-padded FFT that will be taken at each overlap-save FFT.
  The requirement is that `nfft >= size + nDiameter - 1` for both dimensions. If
  `nfft` isn't provided, suitable numbers with small prime factors will be
  selected. If `size` isn't specified, a small multiple of `nDiameter` is
  chosen.
  
  N.B. It is beneficial to make `size` as big as can fit in your system memory.
  Suppose `nDiameter` is 1000. If you make `size=[15*1024, 15*1024]`,
  overlap-save will pick `nfft=[16*1024, 16*1024]` or a bit smaller. A 16k by
  16k array of float64 (actually, they'll be complex128, but the real-only FFT
  will only need half as much space, due to Fourier symmetry) uses 2 GB of
  memory. You'll probably need 4x this much to store all the intermediate
  FFT-related arrays:

  1. the FFT of the spatial filter,
  2. the FFT of the roughly 16k by 16k chunk of input
  3. the product of the two
  4. the inverse-FFT of the product

  I assume your input pixels are int16 or float32, so much smaller before FFT
  than after. So if your system has 8 GB free, you could pick `size=[15*1024,
  15*1024]`. A rough equation might be, if your system has `M` GB, let each
  element of `size` be roughly `np.sqrt(M / 4 * 1024**3 / 8) - nDiameter`.

  `out` allows you to specify the output array to store the results in. This is
  useful when you have a memory-mapped array prepared to accept the output of
  the algorithm, which will be float64. If `out.dtype` is not `float64`, then
  Numpy will perform a conversion, which might be expensive. If provided, this
  is returned. If not specified, a new array is allocated, filled, and returned.
  """
  if filter:
    h = filter
  elif alpha and nDiameter:
    h = makeFilter([nDiameter], alpha, x.dtype)
  else:
    raise ValueError("either (alpha and nDiameter) or filter needed")

  return ols(x, h, size=size, nfft=nfft, out=out)
