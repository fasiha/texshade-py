# -*- coding: utf-8 -*-

import numpy as np
import texshade
import postprocess
from scipy.signal import fftconvolve

fname = 'merged.tif.npy'
arr = np.load(fname)

clip = True
if clip:
  arr = arr[-1500:, -1500:]

alpha = 0.8

postprocess.texToPng(
    texshade.texshadeFFT(arr, alpha),
    'orig-texshade-alpha-{}{}.png'.format(alpha, '-clip' if clip else ''),
    quantiles=[.01, .99],
    borderFractions=[1e-2, 1e-2])

Nwidth = 500
Nhalfband = 128

h = texshade.hankel.halfHankel(Nwidth, alpha, hbTaps=Nhalfband)
print('halfbanded', h.shape)
postprocess.texToPng(
    fftconvolve(arr, h, mode='same'),
    'hankel-texshade-alpha-{}-n-{}{}.png'.format(alpha, Nwidth, '-clip' if clip else ''),
    quantiles=[.01, .99],
    borderFractions=[1e-2, 1e-2])

hFull = texshade.hankel.fullHankel(Nwidth, alpha)
print('non-halfbanded', hFull.shape)
postprocess.texToPng(
    fftconvolve(arr, hFull, mode='same'),
    'hankel-texshadeFullband-alpha-{}-n-{}{}.png'.format(alpha, Nwidth, '-clip' if clip else ''),
    quantiles=[.01, .99],
    borderFractions=[1e-2, 1e-2])
