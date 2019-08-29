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


def fullHankel(n, alpha, samples=None):
  if not samples:
    samples = n * 100
  rvec = np.arange(-n, n)
  rmat = np.sqrt(vec(rvec)**2 + vec(rvec).T**2)

  r = np.linspace(np.sqrt(2) * -n * 1.01, np.sqrt(2) * n * 1.01, samples)
  h = np.array(list(map(lambda x: spatial(x, alpha), r)))
  oned = interp1d(r, h)
  hmat = oned(rmat)
  # hmat = np.reshape(list(map(lambda x: spatial(x, 1.0), rmat.ravel())), rmat.shape)
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
  hbFilter = design(32)
  doubleFilter = convolve2d(
      convolve2d(hmat, vec(hbFilter), mode='same'), vec(hbFilter).T, mode='same')
  n = hmat.shape[0]
  finalFilter = doubleFilter[:-1:2, :-1:2] if n % 4 == 0 else doubleFilter[1:-1:2, 1:-1:2]
  return finalFilter


if __name__ == '__main__':
  import pylab as plt
  plt.ion()

  hmat = fullHankel(150, 0.8)

  plotF2sym(hmat)
  plt.title('Frequency response of full Hankel filter')
  plt.savefig('full-hankel.png', dpi=300)
  plt.savefig('full-hankel.svg', dpi=300)

  finalFilter = halfband(hmat, 32)
  plotF2sym(finalFilter)
  plt.title('Frequency response of half-banded Hankel filter')
  plt.savefig('half-hankel.png', dpi=300)
  plt.savefig('half-hankel.svg', dpi=300)
