# -*- coding: utf-8 -*-
"""
Quick script intended to be used only by a user to convert a specific
GeoTIF to a NPY file for pure-Numpy non-GDAL demo.
"""
import numpy as np
from osgeo import gdal, gdalconst

fname = 'merged.vrt'


def filenameToData(fname: str, dtype=np.float32):
  """Reads all bands"""
  fileHandle = gdal.Open(fname, gdalconst.GA_ReadOnly)
  result = np.squeeze(
      np.dstack(
          [fileHandle.GetRasterBand(n + 1).ReadAsArray() for n in range(fileHandle.RasterCount)]))

  print(f'min={result.min()}, max={result.max()}')
  nodata = [fileHandle.GetRasterBand(n + 1).GetNoDataValue() for n in range(fileHandle.RasterCount)]
  print('no data value:', nodata)
  result[result==np.max(nodata)] = 0
  print(f'new min={result.min()}, max={result.max()}')

  if dtype is not None:
    return result.astype(dtype)
  return result


np.save(fname, filenameToData(fname))
