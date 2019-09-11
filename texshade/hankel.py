# -*- coding: utf-8 -*-

from mpmath import hyper
import numpy as np
from scipy import signal
from scipy.signal import convolve2d
import numpy.fft as fft
import functools
from scipy.interpolate import interp1d


@functools.lru_cache(maxsize=None)
def spatial(r, a, integralMax=np.pi):
  "Evaluate L(r), proportional to the Fourier transform of |f|**α"
  # Wolfram Alpha: `2*pi*Integrate[f^a * BesselJ[0, k * f] * f, f, 0, m]`
  return float(hyper((0.5 * a + 1,), (1.0, 0.5 * a + 2), -0.25 * (r * integralMax)**2))


def vec(v):
  "Convert a Numpy array to a column vector"
  return v.reshape(v.size, -1)


def fullHankel(n: int, alpha: float, interpMethod=True, sampleSpacing=None):
  """Build a FIR filter approximating the fractional-Laplacian operator in the
  middle of its frequency response (non-ideal)

  The returned array will be an `2 * n` by `2 * n` array. `alpha` is the scale
  fraction parameter that governs the amount of sharpening done by the
  fractional-Laplacian operator.

  If `interpMethod=True`, then the values in the filter are
  linearly-interpolated from a grid of samples of the true function (via Hankel
  transform, implemented in `hankel.spatial`), sampled every `sampleSpacing`
  pixels. By default this is the case, and `sampleSpacing` is set to 1e-2
  pixels. This grid is saved to disk.

  If `interpMethod=False`, then each pixel in the output array is exactly
  calculated via `hankel.spatial`. This might be much slower.

  The returned array has a Fourier transform of $|f|^α$ where $|f| < π$ radians
  per sample, and zero otherwise: that is, a circle inscribed in the unit square
  that contains the fractal response surrounded by zeros to the corners. This
  isn't exactly the same as the fractional-Laplacian operator, which is $|f|^α$
  for all $|f|$, even out to the corners.

  See `hankel.halfHankel` for the fully-approximating version of this.
  """
  rvec = np.arange(-n, n)
  rmat = np.sqrt(vec(rvec)**2 + vec(rvec).T**2)

  if interpMethod:
    if not sampleSpacing:
      sampleSpacing = 1e-2
    data = precomputeLoad(alpha, np.ceil(np.sqrt(2) * n * 1.01), sampleSpacing)
    oned = interp1d(data['r'], data['h'])
    hmat = oned(rmat)
  else:
    fun = np.vectorize(lambda x: spatial(x, alpha))
    hmat = fun(rmat)
  return hmat


def designHalfband(N: int, transitionWidth: float):
  """Use the Remez exchange to design a halfband low-pass filter

  `N` taps, with `transitionWidth < 0.5` governing the transition band.
  """
  if N % 2 != 0:
    raise ValueError('N must be even')
  if N < 2:
    raise ValueError('N must be > 1')
  if not (transitionWidth > 0 and transitionWidth < 0.5):
    raise ValueError('Need 0 < transitionWidth < 0.5')
  bands = np.array([0., .25 - transitionWidth, .25 + transitionWidth, .5])
  h = signal.remez(N + 1, bands, [1, 0], [1, 1])
  h[abs(h) <= 1e-4] = 0.0
  return h


def halfband(hmat, taps=128, transitionWidth=0.03):
  """Decimate an array by half

  Design a low-pass halfband filter with `taps` length and with transition band
  `transitionWidth` and apply it to the input array `hmat`, then throw away
  every other sample (2x downsample).
  """
  hbFilter = designHalfband(taps, transitionWidth)
  doubleFilter = convolve2d(
      convolve2d(hmat, vec(hbFilter), mode='same'), vec(hbFilter).T, mode='same')
  n = hmat.shape[0]
  finalFilter = doubleFilter[:-1:2, :-1:2] if n % 4 == 0 else doubleFilter[1:-1:2, 1:-1:2]
  return finalFilter


def halfHankel(n, alpha, interpMethod=True, sampleSpacing=None, hbTaps=128, hbtransitionWidth=0.03):
  """Build the FIR filter approximating the fractional-Laplacian operator over
  all frequencies (ideal)

  Returns an `n` by `n` array representing the spatial-domain FIR filter that
  approximates the `alpha`-detail fractional-Laplacian operator in its total
  frequency response. This should be used instead of `hankel.fullHankel`.

  `interpMethod` and `sampleSpacing` keyword arguments per `hankel.fullHankel`,
  which generates the non-ideal spatial filter.

  A half-band filter of `hbTaps` length and with transition band
  `hbtransitionWidth` is designed (via `hankel.designHalfband`) and used to
  decimate the output of `hankel.fullHankel`. This decimated array is returned.
  """
  return halfband(
      fullHankel(n, alpha, interpMethod=interpMethod, sampleSpacing=sampleSpacing),
      taps=hbTaps,
      transitionWidth=hbtransitionWidth)


def precomputeLoad(alpha: float, N: int, spacing: float):
  """Store and load gridded evaluations of the spatial-domain function
  `hankel.spatial` to disk

  Given `alpha` the detail parameter of the fractional-Laplacian operator, and
  `N` and `spacing`, imagine generating `x = np.arange(0, N, spacing)`. This
  function will return `hankel.spatial` evaluated for each sample of that `x`.

  The data is cached to the current directory, e.g., `hankel-alpha-0.8.npz` for
  `alpha=0.8`. This function will use as much of the pre-computed data from this
  NPZ file as possible, compute whatever is missing from this file, and update
  the file.
  """
  import os.path
  fun = np.vectorize(lambda x: spatial(x, alpha))
  r = np.arange(0, N, spacing)
  fname = 'hankel-alpha-{}'.format(alpha)
  if os.path.isfile(fname + '.npz'):
    npz = np.load(fname + '.npz')
    rsave = npz['x']
    hsave = npz['y']

    rnew = np.sort(np.setdiff1d(r, rsave))
    hnew = fun(rnew) if rnew.size > 0 else []

    r = np.hstack([rsave, rnew])
    h = np.hstack([hsave, hnew])
    idx = np.argsort(r)  # wasteful but resorting 1k vec is fast
    r = r[idx]
    h = h[idx]
  else:
    h = fun(r)
  np.savez(fname, x=r, y=h)
  return dict(r=r, h=h)
