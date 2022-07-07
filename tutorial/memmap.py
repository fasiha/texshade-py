# -*- coding: utf-8 -*-

import numpy as np
from texshade import texshadeSpatial
import postprocess

fname = 'merged.vrt.npy'
arr = np.load(fname, mmap_mode='r')

alpha = 0.8
Nwidth = 500

tex = np.lib.format.open_memmap('mmap-tex.npy', mode='w+', dtype=np.float64, shape=arr.shape)

texshadeSpatial(arr, alpha=alpha, nDiameter=Nwidth, out=tex, size=[1500, 2000])

postprocess.makeGeoTiffLike(
    postprocess.texToImageData(tex, quantiles=[.01, .99], borderFractions=[1e-2, 1e-2]), 'mmap.tif',
    'merged.vrt')
