# -*- coding: utf-8 -*-

import scipy.fftpack as scifft
import numpy as np
from nextprod import nextprod


def texshadeFFT(x, alpha, verbose=True):
  Nyx = [nextprod([2, 3, 5, 7], x) for x in x.shape]

  fy = scifft.rfftfreq(Nyx[0])[:, np.newaxis].astype(x.dtype)
  fx = scifft.rfftfreq(Nyx[1])[np.newaxis, :].astype(x.dtype)
  H2 = (fx**2 + fy**2)**(alpha / 2.0)
  if verbose:
    print("Generated filter")

  rfft2 = lambda x: scifft.rfft(scifft.rfft(x, Nyx[1], 1, True), Nyx[0], 0, True)
  irfft2 = lambda X: scifft.irfft(scifft.irfft(X, axis=0, overwrite_x=True), overwrite_x=True)

  xr = rfft2(x) * H2
  if verbose:
    print("Completed frequency domain operations")
  H2 = None  # potentially trigger GC here to reclaim H2's memory
  xr = irfft2(xr)
  if verbose:
    print("Back to spatial-domain")

  return xr[:x.shape[0], :x.shape[1]]
from ols import ols
from .hankel import halfHankel


def texshadeSpatial(
    x,
    alpha: float,
    # halfHankel args
    nDiameter: int,
    # halfHankel kwargs
    interpMethod=True,
    sampleSpacing=None,
    hbTaps=128,
    hbPassbandWidth=0.03,
    # ols kwargs
    size=None,
    nfft=None,
    out=None,
):

  h = halfHankel(
      nDiameter,
      alpha,
      interpMethod=interpMethod,
      sampleSpacing=sampleSpacing,
      hbTaps=hbTaps,
      hbPassbandWidth=hbPassbandWidth,
  )

  return ols(x, h, size=size, nfft=nfft, out=out)
