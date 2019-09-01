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
  # Wolfram Alpha: `2*pi*Integrate[f^a * BesselJ[0, k * f] * f, f, 0, m]`
  return float(hyper((0.5 * a + 1,), (1.0, 0.5 * a + 2), -0.25 * (r * integralMax)**2))


def vec(v):
  return v.reshape(v.size, -1)


def fullHankel(n, alpha, interpMethod=True, sampleSpacing=None):
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


def design(N=32, passbandWidth=0.03):
  if N % 2 != 0:
    raise ValueError('N must be even')
  if N < 2:
    raise ValueError('N must be > 1')
  if not (passbandWidth > 0 and passbandWidth < 0.5):
    raise ValueError('Need 0 < passbandWidth < 0.5')
  bands = np.array([0., .25 - passbandWidth, .25 + passbandWidth, .5])
  h = signal.remez(N + 1, bands, [1, 0], [1, 1])
  h[abs(h) <= 1e-4] = 0.0
  return h


def halfband(hmat, taps=32):
  hbFilter = design(taps)
  doubleFilter = convolve2d(
      convolve2d(hmat, vec(hbFilter), mode='same'), vec(hbFilter).T, mode='same')
  n = hmat.shape[0]
  finalFilter = doubleFilter[:-1:2, :-1:2] if n % 4 == 0 else doubleFilter[1:-1:2, 1:-1:2]
  return finalFilter


def precomputeLoad(alpha, N, spacing):
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
