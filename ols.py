import numpy as np


def prepareh(h, nfft):
  return np.conj(np.fft.rfftn(h, nfft))


def ols(x, hfftconj, starts, lengths, nfft, nh):
  lengths = np.minimum(np.array(lengths), x.shape - np.array(starts))
  assert np.all(np.array(nfft) >= lengths + np.array(nh) - 1)
  slices = tuple(
      slice(start, start + length + nh - 1) for (start, length, nh) in zip(starts, lengths, nh))
  xpart = x[slices]
  output = np.fft.irfftn(np.fft.rfftn(xpart, nfft) * hfftconj, nfft)
  return output[tuple(slice(0, s) for s in lengths)]
