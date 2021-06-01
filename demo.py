# -*- coding: utf-8 -*-

import texshade
import numpy as np

fname = 'merged.tif.npy'

arr = np.load(fname)
print(arr)
tex = texshade.texshadeFFT(arr, 0.8)
np.save(fname + '.tex', tex)
