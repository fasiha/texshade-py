import numpy as np
from nextprod import nextprod
from arrayRange import arrayRange


def prepareh(h, nfft):
  return np.conj(np.fft.rfftn(h, nfft))


def olsStep(x, hfftconj, starts, lengths, nfft, nh):
  lengths = np.minimum(np.array(lengths), x.shape - np.array(starts))
  assert np.all(np.array(nfft) >= lengths + np.array(nh) - 1)
  slices = tuple(
      slice(start, start + length + nh - 1) for (start, length, nh) in zip(starts, lengths, nh))
  xpart = x[slices]
  output = np.fft.irfftn(np.fft.rfftn(xpart, nfft) * hfftconj, nfft)
  return output[tuple(slice(0, s) for s in lengths)]


def ols(x, h, size=None, nfft=None):
  assert len(x.shape) == len(h.shape)
  size = size or [4 * x for x in h.shape]
  nfft = nfft or [int(nextprod([2, 3, 5, 7], size + nh - 1)) for size, nh in zip(size, h.shape)]
  assert len(x.shape) == len(size)
  assert len(x.shape) == len(nfft)

  hpre = prepareh(h, nfft)
  out = np.zeros(x.shape, dtype=x.dtype)

  for tup in arrayRange([0 for _ in out.shape], out.shape, size):
    out[tup] = olsStep(x, hpre, [s.start for s in tup], size, nfft, h.shape)
  return out
