# -*- coding: utf-8 -*-

import scipy.fftpack as scifft
import numpy as np
from nextprod import nextprod


def texshadeFFT(x, alpha, verbose=False):
  """FFT-based texture-shading elevation

  Given an array `x` of elevation data and an `alpha` > 0, apply the
  texture-shading algorithm using the full (real-only) FFT: the entire `x` array
  will be FFT'd.

  `alpha` is the shading detail factor, i.e., the power of the
  fractional-Laplacian operator. `alpha=0` means no detail (output is the
  input). `alpha=2.0` is the full (non-fractional) Laplacian operator and is
  probably too high. `alpha <= 1.0` seem aesthetically pleasing.

  Returns an array the same dimensions as `x` that contains the texture-shaded
  version of the input array.
  """
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
    hbtransitionWidth=0.03,
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
      hbtransitionWidth=hbtransitionWidth,
  )

  return ols(x, h, size=size, nfft=nfft, out=out)
