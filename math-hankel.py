# -*- coding: utf-8 -*-

import numpy as np
from mpmath import hyper
import numpy.fft as fft
import pylab as plt

plt.style.use('ggplot')


def spatial(r, a):
  "Evaluate L(r), proportional to the Fourier transform of |f|**α"
  return float(hyper((0.5 * a + 1,), (1.0, 0.5 * a + 2), -0.25 * (r * np.pi)**2))


xmat, ymat = np.meshgrid(np.arange(-100, 100), np.arange(-100, 100))
rmat = np.sqrt(xmat**2 + ymat**2)
alpha = 0.8
h = np.vectorize(lambda r: spatial(r, alpha))(rmat)
def F2cent(arr):
  """Origin-centered 2D Fourier transform"""
  return fft.fftshift(fft.fft2(fft.ifftshift(arr)))


def plotF2cent(arr):
  """Given an origin-centered 2D array, plot its 2D Fourier transform"""

  def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta / 2, f[-1] + delta / 2]

  h, w = arr.shape
  x = np.ceil(np.arange(w) - w / 2) / w
  y = np.ceil(np.arange(h) - h / 2) / h
  fig, (sax, fax) = plt.subplots(1, 2)

  sax.imshow(
      arr,
      aspect='equal',
      interpolation='none',
      extent=extents(x * w) + extents(y * h),
      origin='lower')

  fax.imshow(
      np.real(F2cent(arr)),
      aspect='equal',
      interpolation='none',
      extent=extents(x) + extents(y),
      origin='lower')
  sax.grid(False)
  fax.grid(False)
  sax.set_xlabel('pixel')
  fax.set_xlabel('cycles/pixel')

  return fig, sax, fax


hplots = plotF2cent(h)
hplots[1].set_title('L(r): spatial-domain')
hplots[2].set_title('F[L(r)]: frequency-domain')
plt.savefig('full-hankel.png', dpi=300, bbox_inches='tight')
plt.savefig('full-hankel.svg', bbox_inches='tight')

actual = np.real(F2cent(h))[100, :]
f = np.ceil(np.arange(200) - 200 / 2) / 200
expected = np.abs(f * 4)**alpha
plt.figure()
plt.plot(f, actual, '-', f, expected, '--', f, actual / expected, '.')
plt.xlabel('f (cycles/pixel)')
plt.legend(['actual', 'expected', 'actual/expected'])
plt.title('Cut of actual F[L(r)] versus expected |4⋅f|^0.8')
plt.savefig('full-hankel-actual-expected.png', dpi=300, bbox_inches='tight')
plt.savefig('full-hankel-actual-expected.svg', bbox_inches='tight')
import scipy.signal as sig

lpf = sig.iirfilter(8, 0.5, btype='lowpass', ftype='butter')
hiir = sig.filtfilt(*lpf, sig.filtfilt(*lpf, h, axis=0), axis=1)

lpfplots = plotF2cent(hiir)
decplots = plotF2cent(hiir[:-1:2, :-1:2])

lpfplots[1].set_title('LPF[L(r)]')
lpfplots[2].set_title('F[LPF[L(r)]]')
lpfplots[0].savefig('lpf-hankel.png', dpi=300, bbox_inches='tight')
lpfplots[0].savefig('lpf-hankel.svg', bbox_inches='tight')

decplots[1].set_title('HB[L(r)]')
decplots[2].set_title('F[HB[L(r)]]')
decplots[0].savefig('hb-hankel.png', dpi=300, bbox_inches='tight')
decplots[0].savefig('hb-hankel.svg', bbox_inches='tight')
