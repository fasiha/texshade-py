# -*- coding: utf-8 -*-

import numpy as np
from texshade import halfHankel
import postprocess
from ols import ols

fname = 'merged.tif.npy'
arr = np.load(fname, mmap_mode='r')


def texToFile(tex, fname):
  minmax = np.quantile(tex.ravel(), [.01, .99])
  scaled = postprocess.touint(tex, minmax[0], minmax[1], np.uint8)
  postprocess.toPng(scaled, fname)


alpha = 0.8

Nwidth = 500
Nhalfband = 128
h = halfHankel(Nwidth, alpha, hbTaps=Nhalfband)

tex = np.lib.format.open_memmap('mmap-tex.npy', mode='w+', dtype=np.float64, shape=arr.shape)
ols(arr, h, size=[2000, 2000], out=tex)
texToFile(tex, 'hankel-texshade-alpha-{}-n-{}-mmap.png'.format(alpha, Nwidth))
