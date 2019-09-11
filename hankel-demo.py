# -*- coding: utf-8 -*-

import numpy as np
import texshade
import postprocess
from scipy.signal import fftconvolve
nextpow2 = lambda v: list(map(int, 2**np.ceil(np.log2(v))))

fname = 'merged.tif.npy'
arr = np.load(fname)

clip = True
if clip:
  arr = arr[-1500:, -1500:]


def texToFile(tex, fname):
  minmax = np.quantile(tex.ravel(), [.01, .99])
  scaled = postprocess.touint(tex, minmax[0], minmax[1], np.uint8)
  postprocess.toPng(scaled, fname)


alpha = 0.8

texToFile(
    texshade.texshadeFFT(arr, alpha),
    'orig-texshade-alpha-{}{}.png'.format(alpha, '-clip' if clip else ''))

Nwidth = 500
Nhalfband = 128

h = texshade.hankel.halfHankel(Nwidth, alpha, hbTaps=Nhalfband)
print('halfbanded', h.shape)
texToFile(
    fftconvolve(arr, h, mode='same'),
    'hankel-texshade-alpha-{}-n-{}{}.png'.format(alpha, Nwidth, '-clip' if clip else ''))

hFull = texshade.hankel.fullHankel(Nwidth, alpha)
print('non-halfbanded', hFull.shape)
texToFile(
    fftconvolve(arr, hFull, mode='same'),
    'hankel-texshadeFullband-alpha-{}-n-{}{}.png'.format(alpha, Nwidth, '-clip' if clip else ''))
