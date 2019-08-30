# -*- coding: utf-8 -*-

from mpmath import hyper
import numpy as np
from scipy import signal
from scipy.signal import convolve2d
import numpy.fft as fft
import functools
from scipy.interpolate import interp1d


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


@functools.lru_cache(maxsize=None)
def spatial(r, a, integralMax=np.pi):
  # Wolfram Alpha: `2*pi*Integrate[f^a * BesselJ[0, k * f] * f, f, 0, m]`
  return float(hyper((a / 2.0 + 1,), (1.0, a / 2.0 + 2), -0.25 * (r * integralMax)**2))


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


F2sym = lambda arr: fft.fftshift(fft.fft2(fft.ifftshift(arr)))


def plotF2sym(arr):

  def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta / 2, f[-1] + delta / 2]

  h, w = arr.shape
  x = np.ceil(np.arange(w) - w / 2) / w
  y = np.ceil(np.arange(h) - h / 2) / h
  plt.figure()
  plt.imshow(
      np.real(F2sym(arr)),
      aspect='equal',
      interpolation='none',
      extent=extents(x) + extents(y),
      origin='lower')


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


if __name__ == '__main__':
  import pylab as plt
  plt.ion()

  N = 500
  hmat = fullHankel(N, 0.8)

  plotF2sym(hmat)
  plt.title('Frequency response of full Hankel filter')
  plt.savefig('full-hankel.png', dpi=300)
  plt.savefig('full-hankel.svg', dpi=300)

  finalFilter = halfband(hmat, 128)
  plotF2sym(finalFilter)
  plt.title('Frequency response of half-banded Hankel filter')
  plt.savefig('half-hankel.png', dpi=300)
  plt.savefig('half-hankel.svg', dpi=300)

  Hf = np.real(F2sym(hmat))
  h, w = Hf.shape
  x = np.ceil(np.arange(w) - w / 2) / w
  x = x / .5 * 2
  plt.figure()
  plt.plot(x, Hf[N, :], x, x**.8)

  H2f = np.real(F2sym(finalFilter))
  h2, w2 = H2f.shape
  x2 = np.ceil(np.arange(w2) - w2 / 2) / w2
  x2 = x2 / .5 * 2
  remmax = lambda x: x / np.max(x)
  plt.figure()
  plt.plot(x2, H2f[N // 2, :], x2,
           np.abs(x2)**.8, x2,
           remmax(np.abs(x2)**.8) * np.max(H2f[N // 2, :]))
