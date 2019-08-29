# -*- coding: utf-8 -*-

import pylab as plt
plt.ion()
from mpmath import hyper
import numpy as np
from scipy import signal
from scipy.signal import convolve2d, convolve
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


rvec = np.arange(-150, 150)
rmat = np.sqrt(vec(rvec)**2 + vec(rvec).T**2)

r = np.linspace(np.sqrt(2) * -150 * 1.01, np.sqrt(2) * 150 * 1.01, 10000)
h = np.array(list(map(lambda x: spatial(x, 0.8), r)))
oned = interp1d(r, h)
hmat = oned(rmat)
# hmat = np.reshape(list(map(lambda x: spatial(x, 1.0), rmat.ravel())), rmat.shape)

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


plotF2sym(hmat)
plt.title('Frequency response of full Hankel filter')

hbFilter = design(32)
doubleFilter = convolve2d(
    convolve2d(hmat, vec(hbFilter), mode='same'), vec(hbFilter).T, mode='same')
finalFilter = doubleFilter[:-1:2, :-1:2] if r.size % 4 == 0 else doubleFilter[1:-1:2, 1:-1:2]

plotF2sym(finalFilter)
plt.title('Frequency response of half-banded Hankel filter')
