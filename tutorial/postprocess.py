# -*- coding: utf-8 -*-

import numpy as np
from osgeo import gdal, gdalconst, osr


def touint(x: np.ndarray, cmin, cmax, dtype=np.uint8) -> np.ndarray:
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


def toPng(scaled: np.ndarray, fname: str):
  """Write a uint8 array `scaled` to a PNG file `fname`"""
  from PIL import Image
  newimage = Image.new('L', (scaled.shape[1], scaled.shape[0]))  # type, (width, height)
  newimage.putdata(scaled.ravel())
  newimage.save(fname)


def texToPng(tex: np.ndarray, fname: str, quantiles=None, borderFractions=None):
  """Quantile a texture-shaded array and write it to 8-bit PNG

  Given `tex`, a 2D array, and a `fname` path to a PNG file, and optionally a
  2-list `quantiles` (defaults to [0.01, 0.99], i.e., 1% and 99%), clamp the
  array to the quantile-values and write to a PNG. If `borderFractions`, also a
  2-list, is given, 

  `[np.round(total * frac) for total, frac in zip(tex.shape, borderFractions)]`
  
  pixels on either side of the border in each dimension are ignored in
  computing the quantiles.
  """
  if quantiles is None:
    quantiles = [0.01, 0.99]
  assert all([x >= 0 and x <= 1 for x in quantiles])
  if borderFractions is None:
    minmax = np.quantile(tex.ravel(), quantiles)
  else:
    assert all([x >= 0 and x < 1 for x in borderFractions])
    border = [int(np.round(total * frac)) for total, frac in zip(tex.shape, borderFractions)]
    slices = tuple(slice(p, -p if p > 0 else None) for p in border)
    minmax = np.quantile(tex[slices].ravel(), quantiles)

  scaled = touint(tex, minmax[0], minmax[1], np.uint8)
  toPng(scaled, fname)
  return scaled


# Adapted from EddyTheB, http://gis.stackexchange.com/a/57006/8623
def getGeoInfo(filename):
  "Extract a bunch of GDAL-specific metadata from a file"
  source = gdal.Open(filename, gdalconst.GA_ReadOnly)
  noDataValue = source.GetRasterBand(1).GetNoDataValue()
  xsize = source.RasterXSize
  ysize = source.RasterYSize
  geoTransform = source.GetGeoTransform()
  proj = osr.SpatialReference()
  proj.ImportFromWkt(source.GetProjectionRef())
  dtype = source.GetRasterBand(1).DataType
  dtype = gdal.GetDataTypeName(dtype)
  return noDataValue, xsize, ysize, geoTransform, proj, dtype


def createGeoTiff(filename,
                  array,
                  driver,
                  noDataValue,
                  xsize,
                  ysize,
                  geoTransform,
                  proj,
                  dtype,
                  numBands=1):
  "Given an array, and a bunch of GDAL metadata, create a GeoTIFF"
  # Set up the dataset
  DataSet = driver.Create(filename, xsize, ysize, numBands, dtype, options=['COMPRESS=LZW'])
  DataSet.SetGeoTransform(geoTransform)
  DataSet.SetProjection(proj.ExportToWkt())
  # Write the array
  if numBands == 1:
    DataSet.GetRasterBand(numBands).WriteArray(array)
    if noDataValue is not None:
      DataSet.GetRasterBand(numBands).SetNoDataValue(noDataValue)
  else:
    for bid in range(numBands):
      DataSet.GetRasterBand(bid + 1).WriteArray(array[:, :, bid])
      if noDataValue is not None:
        DataSet.GetRasterBand(bid + 1).SetNoDataValue(noDataValue)
  return filename


if __name__ == '__main__':
  tex = np.load('merged.tif.npy.tex.npy')
  scaled = texToPng(tex, 'scaled.png', quantiles=[.01, .99], borderFractions=[1e-2, 1e-2])

  # save as GeoTiff
  driver = gdal.GetDriverByName('GTiff')
  noDataValue, xsize, ysize, geoTransform, proj, dtype = getGeoInfo('merged.tif')
  createGeoTiff('scaled.tif', scaled, driver, 0, xsize, ysize, geoTransform, proj,
                gdalconst.GDT_Byte)
  print('done exporting texshaded PNG and GeoTIFF')

  # write original DEM too
  tex = np.load('merged.tif.npy')
  toPng(touint(tex, np.min(tex), np.max(tex), np.uint8), 'orig.png')
