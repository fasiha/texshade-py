# -*- coding: utf-8 -*-

import numpy as np


def touint(x, cmin, cmax, dtype=np.uint8):
  """Convert an array to an array of unsigned integers by clamping and scaling

  Given an array of numbers `x`, and the desired min and max values, `cmin` and
  `cmax` respectively, and optionally a `dtype` that defaults to `uint8`, clamp
  the values of `x` to between `cmin` and `cmax` (i.e., if a pixel is less than
  `cmin`, it will be treated as being equal to `cmin`) and scale the values
  linearly to the full range supported by `dtype`. When `dtype` is `np.uint8`,
  e.g., the output will have values between 0 (originally `cmin`) and 255
  (originally `cmax`).
  """
  # clamp x between cmin and cmax
  x[x < cmin] = cmin
  x[x > cmax] = cmax
  # map [cmin, cmax] to [0, 2**depth-1-eps] linearly
  maxval = 2**(8 * dtype().itemsize) - 1e-3
  slope = (maxval - 1.0) / (cmax - cmin)
  ret = slope * (x - cmin) + 1
  return (ret).astype(dtype)


def toPng(scaled, fname: str):
  """Write a uint8 array `scaled` to a PNG file `fname`"""
  from PIL import Image
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
