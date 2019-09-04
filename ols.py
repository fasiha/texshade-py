import numpy as np


def prepareh(h, nfft):
  return np.conj(np.fft.rfft(h, nfft))


def ols(x, hfftconj, rangeObj, nfft, nh):
  start = rangeObj[0]
  length = min(max(rangeObj) + 1, x.size) - start
  assert (nfft >= length + nh - 1)
  xpart = x[start:(start + length + nh - 1 + 1 - 1)]
  output = np.fft.irfft(np.fft.rfft(xpart, nfft) * hfftconj)
  return output[:length]


if __name__ == '__main__':
  x = np.random.randn(100)
  h = np.random.randn(8)
  ngold = x.size + h.size - 1
  gold = np.real(np.fft.ifft(np.fft.fft(x, ngold) * np.conj(np.fft.fft(h, ngold)))[:x.size])
  nfft = 10
  preh = prepareh(h, nfft)
  for chunks in range(1, x.size + 1):
    v = np.hstack(
        [ols(x, preh, range(start, start + 3), nfft, h.size) for start in range(0, x.size, 3)])
    assert (np.allclose(v, gold))
