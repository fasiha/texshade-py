# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image


def touint(x, cmin, cmax, dtype=np.uint8):
  # clamp x between cmin and cmax
  x[x < cmin] = cmin
  x[x > cmax] = cmax
  # map [cmin, cmax] to [0, 2**depth-1-eps] linearly
  maxval = 2**(8 * dtype().itemsize) - 1e-3
  slope = (maxval - 1.0) / (cmax - cmin)
  ret = slope * (x - cmin) + 1
  return (ret).astype(dtype)


def toPng(scaled, fname):
  newimage = Image.new('L', (scaled.shape[1], scaled.shape[0]))  # type, (width, height)
  newimage.putdata(scaled.ravel())
  newimage.save(fname)


if __name__ == '__main__':
  arr = np.load('merged.tif.npy')
  tex = np.load('merged.tif.npy.tex.npy')
  minmax = np.quantile(tex.ravel(), [.01, .99])
  scaled = touint(tex, minmax[0], minmax[1], np.uint8)
  toPng(scaled, 'scaled.png')
  toPng(touint(arr, np.min(arr), np.max(arr), np.uint8), 'orig.png')
