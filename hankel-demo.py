# -*- coding: utf-8 -*-

import numpy as np
import hankel
import postprocess
import numpy.fft as fft

nextpow2 = lambda v: list(map(int, 2**np.ceil(np.log2(v))))

fname = 'merged.tif.npy'
arr = np.load(fname)
Xf = fft.rfft2(arr, nextpow2(np.array(arr.shape) + 1000))

h = hankel.halfband(hankel.fullHankel(1000, 0.8), 64)
# tex = convolve2d(arr, h, mode='same')
Hf = fft.rfft2(h, Xf.shape)
tex = fft.irfft2(Xf * np.conj(Hf))

minmax = np.quantile(tex.ravel(), [.01, .99])
scaled = postprocess.touint(tex, minmax[0], minmax[1], np.uint8)
postprocess.toPng(scaled, '4hankel-texshade.png')
