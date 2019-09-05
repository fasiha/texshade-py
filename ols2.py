import numpy as np


def prepareh(h, nfft):
  return np.conj(np.fft.rfftn(h, nfft))


def ols(x, hfftconj, starts, lengths, nfft, nh):
  lengths = np.minimum(np.array(lengths), x.shape - np.array(starts))
  assert np.all(np.array(nfft) >= lengths + np.array(nh) - 1)
  slices = tuple(
      slice(start, start + length + nh - 1) for (start, length, nh) in zip(starts, lengths, nh))
  xpart = x[slices]
  output = np.fft.irfftn(np.fft.rfftn(xpart, nfft) * hfftconj)
  return output[tuple(slice(0, s) for s in lengths)]


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

  for xshift in range(8):
    for yshift in range(8):
      ystep, xstep = 1 + yshift, 1 + xshift
      dirt = np.vstack([
          np.hstack([
              ols(x, hpre, [ystart, xstart], [ystep, xstep], nfft, h.shape)
              for xstart in range(0, x.shape[0], xstep)
          ])
          for ystart in range(0, x.shape[1], ystep)
      ])
      assert np.allclose(dirt, gold)
  print('success')
