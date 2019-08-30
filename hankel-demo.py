# -*- coding: utf-8 -*-

import numpy as np
import hankel
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
    texshade.texshade(arr, alpha),
    'orig-texshade-alpha-{}{}.png'.format(alpha, '-clip' if clip else ''))

Nwidth = 500
Nhalfband = 128

h = hankel.halfband(hankel.fullHankel(Nwidth, alpha), Nhalfband)
texToFile(
    fftconvolve(arr, h, mode='same'),
    'hankel-texshade-alpha-{}-n-{}{}.png'.format(alpha, Nwidth, '-clip' if clip else ''))
