# -*- coding: utf-8 -*-

import texshade
import numpy as np

fname = 'merged.vrt.npy'

arr = np.load(fname)
tex = texshade.texshadeFFT(arr, 0.8)
np.save(fname + '.tex', tex)
