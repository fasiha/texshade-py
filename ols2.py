import numpy as np


def prepareh(h, nfft):
  return np.conj(np.fft.rfft2(h, nfft))


def ols(x, hfftconj, range0, range1, nfft, nh):
  start0 = range0[0]
  start1 = range1[0]
  length0 = min(max(range0) + 1, x.shape[0]) - start0
  length1 = min(max(range1) + 1, x.shape[1]) - start1
  length = np.array([length0, length1])
  assert np.all(nfft >= length + nh - 1)
  xpart = x[start0:(start0 + length0 + nh[0] - 1 + 1 - 1), start1:(start1 + length1 + nh[1] - 1 +
                                                                   1 - 1)]
  output = np.fft.irfft2(np.fft.rfft2(xpart, nfft) * hfftconj)
  return output[:length0, :length1]


if __name__ == '__main__':
  x = np.arange(-50, 50).reshape([10, 10]) + 0.0
  h = np.arange(-2, 7).reshape([3, 3]) + 0.0
  ngold = np.array(x.shape) + np.array(h.shape) - 1
  gold = np.real(np.fft.ifft2(np.fft.fft2(x, ngold) *
                              np.conj(np.fft.fft2(h, ngold))))[:x.shape[0], :x.shape[1]]
  real = np.fft.irfft2(np.fft.rfft2(x, ngold) *
                       np.conj(np.fft.rfft2(h, ngold)))[:x.shape[0], :x.shape[1]]
  assert (np.allclose(real, gold))
  
  nfft = [10, 12]
  hpre = prepareh(h, nfft)
  
  ystep = 7
  xstep = 5
  col = []
  for ystart in range(0, x.shape[1], ystep):
    row = np.hstack([
        ols(x, hpre, range(ystart, ystart + ystep), range(xstart, xstart + xstep), nfft, h.shape)
        for xstart in range(0, x.shape[0], xstep)
    ])
    col.append(row)
  dirt = np.vstack(col)
  assert np.allclose(dirt, gold)
