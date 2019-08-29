# -*- coding: utf-8 -*-

import texshade
import postprocess
import numpy as np
fname = 'merged.tif.npy'

arr = np.load(fname)

tex = texshade.texshade(arr, 0.8)
minmax = np.quantile(tex.ravel(), [.01, .99])
scaled = postprocess.touint(tex, minmax[0], minmax[1], np.uint8)
postprocess.toPng(scaled, 'scaled-0.8.png')

tex = texshade.texshade(arr, 0.4)
minmax = np.quantile(tex.ravel(), [.01, .99])
scaled = postprocess.touint(tex, minmax[0], minmax[1], np.uint8)
postprocess.toPng(scaled, 'scaled-0.4.png')
