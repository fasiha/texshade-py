import numpy as np
from scipy.signal import fftconvolve
from ols import prepareh, ols
from nextprod import nextprod


def testOls():

  def testouter(nx, nh):
    x = np.random.randint(-30, 30, size=(nx, nx))
    h = np.random.randint(-20, 20, size=(nh, nh))
    ngold = np.array(x.shape) + np.array(h.shape) - 1
    gold = np.real(np.fft.ifft2(np.fft.fft2(x, ngold) *
                                np.conj(np.fft.fft2(h, ngold))))[:x.shape[0], :x.shape[1]]
    real = np.fft.irfft2(np.fft.rfft2(x, ngold) * np.conj(np.fft.rfft2(h, ngold)),
                         ngold)[:x.shape[0], :x.shape[1]]

    assert (np.allclose(real, gold))

    conv = fftconvolve(x, h[::-1, ::-1], 'same')
    conv = np.roll(conv, [-(nh // 2)] * 2, [-1, -2])
    assert (np.allclose(conv[:-(nh // 2), :-(nh // 2)], gold[:-(nh // 2), :-(nh // 2)]))

    def testinner(maxlen):
      nfft = [nextprod([2, 3], x) for x in np.array(h.shape) + maxlen - 1]
      hpre = prepareh(h, nfft)

      for xlen0 in range(maxlen):
        for ylen0 in range(maxlen):
          ylen, xlen = 1 + ylen0, 1 + xlen0
          dirt = np.vstack([
              np.hstack([
                  ols(x, hpre, [ystart, xstart], [ylen, xlen], nfft, h.shape)
                  for xstart in range(0, x.shape[0], xlen)
              ])
              for ystart in range(0, x.shape[1], ylen)
          ])
          assert np.allclose(dirt, gold)

    testinner(8)
    testinner(5)

  for nx in [10, 11, 12, 13]:
    for nh in [3, 4, 5, 6]:
      testouter(nx, nh)
