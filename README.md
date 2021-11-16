# Texshade: texture-shaded elevation via the fractional-Laplacian operator

## Introduction

See [textureshading.com](http://www.textureshading.com/Home.html) for links to papers and slides by Leland Brown from 2010 and 2014 describing the technique of texture shading in detail, but in summary, it is a method of processing digital elevation maps (DEMs) that highlights the network nature of topography, throwing ridges, canyons, and valleys into sharp relief.

My blog post contains some pretty images, mashing up various basemaps with texture-shaded terrain: [Texture-shaded Globe](https://fasiha.github.io/post/texshade/).

This repository contains an open-source public-domain Python/Numpy software library to apply the texture shading algorithm on *extremely* large datasets. This is a challenge because a straightforward implementation of the texture-shading technique requires loading the entire elevation map into memory. For large datasets—like the ASTER Global DEM, which comes in at roughly 250 GB compressed—you either have to find a computer with a lot of memory (nodes with 1+ TB RAM are available at many scientific organizations as of 2019) or you have to modify the technique.

This repository contains (1) mathematical and (2) software details of a low-memory approximation to the original texture-shading algorithm that in practice produces texture-shaded imagery very similar to the full algorithm.

> (The trick is a well-known result from signal processing theory. Rather than building the full filter in the frequency domain, we build a smaller version of it in the spatial domain, and then use the overlap-save method of fast-convolution.) The mathematical trick, in a nutshell, is to use the Hankel transform to find a finite impulse response (FIR) filter that approximates the frequency-domain fractional-Laplacian operator, and apply that filter in the spatial domain via the efficient overlap-save algorithm. According to old Git commit logs, I first derived this technique in 2015.

Links:
- this mathematical–software document for reading: [homepage](https://fasiha.github.io/texshade-py/)
- this code repository: on [GitHub](https://github.com/fasiha/texshade-py/)
- this module: on [PyPI](https://pypi.org/project/texshade/)

## Installation and usage
To install this library:
```
$ pip install texshade
```
To use it, in your Python code:
```py
import texshade
```

The rest of this document provides examples on using the following two major functions exposed by this library. Their API is as follows.

### `def texshadeFFT(x: np.ndarray, alpha: float) -> np.ndarray` FFT-based texture shading elevation

Given an array `x` of elevation data and an `alpha` > 0, apply the texture-shading algorithm using the full (real-only) FFT: the entire `x` array will be FFT'd.

`alpha` is the shading detail factor, i.e., the power of the fractional-Laplacian operator. `alpha=0` means no detail (output is the input). `alpha=2.0` is the full (non-fractional) Laplacian operator and is probably too high. `alpha <= 1.0` seem aesthetically pleasing.

Returns an array the same dimensions as `x` that contains the texture-shaded version of the input array.

If `x` is memory-mapped and/or your system doesn't have 5x `x`'s memory available, consider using `texshade.texshadeSpatial`, which implements a low-memory version of the algorithm by approximating the frequency response of the fractional-Laplacian filter with a finite impulse response filter applied in the spatial-domain.

Implementation note: this function uses Scipy's FFTPACK routines (in `scipy.fftpack`) instead of Numpy's FFT (`numpy.fft`) because the former can return single-precision float32. In newer versions of Numpy/Scipy, this advantage may have evaporated [1](https://github.com/numpy/numpy/issues/6012), [2](https://github.com/scipy/scipy/issues/2487).

### `texshadeSpatial` Low-memory approximation of the texture shading algorithm
Full signature:
```py
def texshadeSpatial(
    x: np.ndarray,
    alpha: float,
    # halfHankel args
    nDiameter: int,
    # halfHankel kwargs
    interpMethod=True,
    sampleSpacing=None,
    hbTaps=128,
    hbTransitionWidth=0.03,
    # ols kwargs
    size=None,
    nfft=None,
    out=None,
) -> np.ndarray:
```

Unlike `texshade.texshadeFFT`, which computes an FFT of the entire input elevation array `x` and applies the fractional-Laplacian filter in the frequency domain, this function approximates that frequency response with a spatial-domain finite impulse response (FIR) filter that is applied in the spatial domain via fast-convolution (overlap-save method). This allows `x` to be memory-mapped and/or very large relative to the amount of free system memory.

`alpha` is the shading detail factor, i.e., the power of the
fractional-Laplacian operator. `alpha=0` means no detail (output is the
input). `alpha=2.0` is the full (non-fractional) Laplacian operator and is
probably too high. `alpha <= 1.0` seem aesthetically pleasing.

Returns an array the same dimensions as `x` that contains the texture-shaded
version of the input array.

`nDiameter` specifies the size of the spatial-domain FIR filter to apply to `x`. It is in the same units as `x`. The larger this is, the closer the output will be to the return value of `texshade.texshadeFFT`. This number controls the size of the neighborhood around a given pixel that contribute to that pixel's final texture-shaded value. If this is too small, the output will differ significantly from the full texture shading algorithm. If it is too big, you may run out of memory, because the overlap-save algorithm for fast-convolution will compute FFTs *at least* this size.

**Spatial filter generation keyword args** passed to `texshade.hankel.halfHankel`: see that function's docstring for details, but reasonable values are chosen for these:

- `interpMethod`
- `sampleSpacing`
- `hbTaps`
- `hbTransitionWidth`

**Overlap-save keyword args** passed to `ols.ols` (this function is in the `overlap-save` module on PyPI):

- `size`
- `nfft`
- `out`

`size`, a 2-list, specifies the size of the sub-arrays of the texture-shaded output to compute in each overlap-save step, while `nfft` (also a 2-list) is the size of the zero-padded FFT that will be taken at each overlap-save FFT. The requirement is that `nfft >= size + nDiameter - 1` for both dimensions. If `nfft` isn't provided, suitable numbers with small prime factors will be selected. If `size` isn't specified, a small multiple of `nDiameter` is chosen.

N.B. It is beneficial to make `size` as big as can fit in your system memory. Suppose `nDiameter` is 1000. If you make `size=[15*1024, 15*1024]`, overlap-save will pick `nfft=[16*1024, 16*1024]` or a bit smaller. A 16k by 16k array of float64 (actually, they'll be complex128, but the real-only FFT will only need half as much space, due to Fourier symmetry) uses 2 GB of memory. You'll probably need 4x this much to store all the intermediate FFT-related arrays:

1. the FFT of the spatial filter,
2. the FFT of the roughly 16k by 16k chunk of input
3. the product of the two
4. the inverse-FFT of the product

I assume your input pixels are int16 or float32, so much smaller before FFT than after. So if your system has 8 GB free, you could pick `size=[15*1024, 15*1024]`. A rough equation might be, if your system has `M` GB, let each element of `size` be roughly `np.sqrt(M / 4 * 1024**3 / 8) - nDiameter`.

`out` allows you to specify the output array to store the results in. This is useful when you have a memory-mapped array prepared to accept the output of the algorithm, which will be float64. If `out.dtype` is not `float64`, then Numpy will perform a conversion, which might be expensive. If provided, this is returned. If not specified, a new array is allocated, filled, and returned.

## The texture-shading algorithm

> N.B. The following contains LaTeX-typeset mathematics. If you see gibberish instead of math, make sure you're reading this on [the repo website](https://fasiha.github.io/texshade-py), where KaTeX will format it (assuming you have JavaScript enabled). And in case you want it, here's the [GitHub repo](https://github.com/fasiha/texshade-py/) itself.

The original texture shading algorithm takes a 2D array of elevations, call it \\(x\\), and computes the texture-shaded elevation map,

$$y = F^{-1}[F[x] ⋅ |\vec f|^α],$$

where
- \\(F[\cdot]\\) is the 2D Fourier transform operator and \\(F^{-1}[\cdot]\\) its inverse
- \\(\vec f = [f_x, f_y]'\\) the 2D vector of Fourier coordinates, so \\(|\vec f|^α=(f_x^2 + f_y^2)^{α/2}\\)
- \\(()'\\) indicates matrix or vector transpose
- \\(0<α≤1\\), the "fraction" in the fractional-Laplacian (though Brown gives examples of \\(α≤2\\)!).

While many details of the algorithm have yet to be specified, the output array \\(y\\) can be made to have the same size as the input \\(x\\).

Let's implement this in Python.

```py
# export texshade/texshade.py
import scipy.fftpack as scifft
import numpy as np
from nextprod import nextprod


def texshadeFFT(x: np.ndarray, alpha: float) -> np.ndarray:
  """FFT-based texture shading elevation

  Given an array `x` of elevation data and an `alpha` > 0, apply the
  texture-shading algorithm using the full (real-only) FFT: the entire `x` array
  will be FFT'd.

  `alpha` is the shading detail factor, i.e., the power of the
  fractional-Laplacian operator. `alpha=0` means no detail (output is the
  input). `alpha=2.0` is the full (non-fractional) Laplacian operator and is
  probably too high. `alpha <= 1.0` seem aesthetically pleasing.

  Returns an array the same dimensions as `x` that contains the texture-shaded
  version of the input array.

  If `x` is memory-mapped and/or your system doesn't have 5x `x`'s memory
  available, consider using `texshade.texshadeSpatial`, which implements a
  low-memory version of the algorithm by approximating the frequency response of
  the fractional-Laplacian filter with a finite impulse response filter applied
  in the spatial-domain.

  Implementation note: this function uses Scipy's FFTPACK routines (in
  `scipy.fftpack`) instead of Numpy's FFT (`numpy.fft`) because the former can
  return single-precision float32. In newer versions of Numpy/Scipy, this
  advantage may have evaporated [1], [2].

  [1] https://github.com/numpy/numpy/issues/6012
  [2] https://github.com/scipy/scipy/issues/2487
  """
  Nyx = [nextprod([2, 3, 5, 7], x) for x in x.shape]

  # Generate filter in the frequency domain
  fy = scifft.rfftfreq(Nyx[0])[:, np.newaxis].astype(x.dtype)
  fx = scifft.rfftfreq(Nyx[1])[np.newaxis, :].astype(x.dtype)
  H2 = (fx**2 + fy**2)**(alpha / 2.0)

  # Define forward and backwards transforms
  rfft2 = lambda x: scifft.rfft(scifft.rfft(x, Nyx[1], 1, True), Nyx[0], 0, True)
  irfft2 = lambda X: scifft.irfft(scifft.irfft(X, axis=0, overwrite_x=True), overwrite_x=True)

  # Compute the FFT of the input and apply the filter
  xr = rfft2(x) * H2
  H2 = None  # potentially trigger GC here to reclaim H2's memory
  xr = irfft2(xr)
  # Return the same size as input
  return xr[:x.shape[0], :x.shape[1]]
```

## Test setup
This section shows the entire pipeline that texture shading is a part of:
- getting some raw digital elevation data as multiple files,
- merging them into a single file,
- applying the texture shading algorithm,
- quantizing the results so each pixel is a byte (256 levels), and
- emitting a georegistered texture-shaded terrain.

This section makes use of GDAL command-line tools, as well as GDAL and Pillow Python libraries, to demonstrate this entire workflow but these tools are *not necessary* to use the library!

Let's begin.

I've downloaded three tiles from the SRTM DEM (from [this page at SDSC.edu](https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/SRTM_GL1/SRTM_GL1_srtm/North/North_0_29/)) on the African coastline near 0° N and 0° W and merged them into a single raster using [GDAL](https://gdal.org/), which I installed using [Brew](https://formulae.brew.sh/formula/gdal): installing these is outside the scope of this document, but any DEM you have can be used.
```
wget https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/SRTM_GL1/SRTM_GL1_srtm/North/North_0_29/N00E009.hgt \
  https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/SRTM_GL1/SRTM_GL1_srtm/North/North_0_29/N00E010.hgt \
  https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/SRTM_GL1/SRTM_GL1_srtm/North/North_0_29/N00E011.hgt
gdalwarp -of GTiff N00E009.hgt N00E010.hgt N00E011.hgt merged.tif
```

Running `gdalinfo merged.tif` produces the following output:
```
Driver: GTiff/GeoTIFF
Files: merged.tif
Size is 10801, 3601
Coordinate System is:
GEOGCS["WGS 84",
    DATUM["WGS_1984",
        SPHEROID["WGS 84",6378137,298.257223563,
            AUTHORITY["EPSG","7030"]],
        AUTHORITY["EPSG","6326"]],
    PRIMEM["Greenwich",0],
    UNIT["degree",0.0174532925199433],
    AUTHORITY["EPSG","4326"]]
Origin = (8.999861111111111,1.000138888888889)
Pixel Size = (0.000277777777778,-0.000277777777778)
Metadata:
  AREA_OR_POINT=Point
Image Structure Metadata:
  INTERLEAVE=BAND
Corner Coordinates:
Upper Left  (   8.9998611,   1.0001389) (  8d59'59.50"E,  1d 0' 0.50"N)
Lower Left  (   8.9998611,  -0.0001389) (  8d59'59.50"E,  0d 0' 0.50"S)
Upper Right (  12.0001389,   1.0001389) ( 12d 0' 0.50"E,  1d 0' 0.50"N)
Lower Right (  12.0001389,  -0.0001389) ( 12d 0' 0.50"E,  0d 0' 0.50"S)
Center      (  10.5000000,   0.5000000) ( 10d30' 0.00"E,  0d30' 0.00"N)
Band 1 Block=10801x1 Type=Int16, ColorInterp=Gray
  NoData Value=-32768
  Unit Type: m
```
This looks good: we have a 10801 by 3601 image whose center is close to the equator, as eppected.

I want to confine GDAL and geo-registered images to the edges of my workflow so I want to convert this elevation data to a simple Numpy array. This next script, `convert.py`, does that.

```py
# export convert.py
"""
Quick script intended to be used only by a user to convert a specific
GeoTIF to a NPY file for pure-Numpy non-GDAL demo.
"""
import numpy as np
import gdal, gdalconst

fname = 'merged.tif'


def filenameToData(fname: str, dtype=np.float32):
  """Reads all bands"""
  fileHandle = gdal.Open(fname, gdalconst.GA_ReadOnly)
  result = np.squeeze(
      np.dstack(
          [fileHandle.GetRasterBand(n + 1).ReadAsArray() for n in range(fileHandle.RasterCount)]))
  if dtype is not None:
    return result.astype(dtype)
  return result


np.save(fname, filenameToData(fname))
```

Now I'd like to apply the texture-shading algorithm—what you've all come for! This script, `demo.py`, exercises the `texshade` library published by this repo. We've picked α of 0.8.
```py
# export demo.py
import texshade
import numpy as np

fname = 'merged.tif.npy'

arr = np.load(fname)
print(arr)
tex = texshade.texshadeFFT(arr, 0.8)
np.save(fname + '.tex', tex)
```

We need a big script to export the texture-shaded Numpy array to a georegistered image, so we can easily compare the output with our usual GIS tools. We'd also like to export the original and texture-shaded terrains as PNG files for easy visualization in browsers. This final script, `postprocess.py`, does all this. I've included it fully for completeness.
```py
# export postprocess.py
import numpy as np
import gdal, gdalconst
from osgeo import osr


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
```

This next command resizes the output images so I can include them in this repo.
```
for i in orig.png scaled.png; do convert -filter Mitchell -sampling-factor 1x1 -quality 90 -resize 2048 $i $i.small.png; done
```

### Original
![original downsampled](orig.png.small.png)

### Tex-shaded
![tex-shaded downsampled](scaled.png.small.png)



## Spatial filtering and fast-convolution for low-memory usage
When we called `texshadeFFT` above, Python computed the two-dimensional FFT of the *entire* elevation array. This means that your computer had enough memory to store 
1. not just the entire elevation array but also
2. its (real-only) Fourier transform,
3. the fractional-Laplacian frequency-domain filter $$|\vec f|^α$$.

Even if we didn't store #3 above (e.g., if we used Numba to modify #2 in-place), since FFTW cannot do in-place Fourier transforms, we're still left with needing 3× the entire elevation array in free memory. 


The textbook definition of convolving two signals in the spatial domain is a quadratic \\(O(N^2)\\) operation. Since convolution in the spatial domain is mathematically equivalent to multiplication in the frequency domain, and the FFT is a log-linear \\(O(N \log N)\\) operation, this is usually much faster—this is why we use `fftconvolve` above. The drawback of the FFT-based alternative to direct convolution is that it requires we run the FFT on the signals of interest—with a potentially prohibitive memory burden.

The overlap-save method (and its closely-related sibling, the overlap-add method) allow us to convolve signals more intelligently: it still uses FFTs, so the overall theoretical runtime complexity remains log-linear, but it uses *many small* FFTs so memory consumption remains reasonable. I prefer overlap-save because it partitions the *output* array into non-overlapping segments that each step of the algorithm fills in (and which may be parallelized). Each step of the overlap-save algorithm reaches for segments of *input* that may overlap with other steps, but this overlap is read-only.

> In contrast, overlap-add splits the *input* array into non-overlapping segments. Each step of that algorithm has to potentially modify previously-computed samples of the *output*, which makes parallelization much more nasty (requiring locks or careful orchestration of the sequence of steps).

The overlap-save implementation I wrote is largely out of the scope of this texture shading library, so let's just import it and show how we can use it, along with memory-mapped inputs and outputs to *really* save memory.

```py
# export texshade/texshade.py
from ols import ols
from .hankel import halfHankel


def texshadeSpatial(
    x: np.ndarray,
    alpha: float,
    # halfHankel args
    nDiameter: int,
    # halfHankel kwargs
    interpMethod=True,
    sampleSpacing=None,
    hbTaps=128,
    hbTransitionWidth=0.03,
    # ols kwargs
    size=None,
    nfft=None,
    out=None,
) -> np.ndarray:
  """Low-memory approximation of the texture shading algorithm

  Unlike `texshade.texshadeFFT`, which computes an FFT of the entire input
  elevation array `x` and applies the fractional-Laplacian filter in the
  frequency domain, this function approximates that frequency response with a
  spatial-domain finite impulse response (FIR) filter that is applied in the
  spatial domain via fast-convolution (overlap-save method). This allows `x` to
  be memory-mapped and/or very large relative to the amount of free system
  memory.

  `alpha` is the shading detail factor, i.e., the power of the
  fractional-Laplacian operator. `alpha=0` means no detail (output is the
  input). `alpha=2.0` is the full (non-fractional) Laplacian operator and is
  probably too high. `alpha <= 1.0` seem aesthetically pleasing.

  Returns an array the same dimensions as `x` that contains the texture-shaded
  version of the input array.

  `nDiameter` specifies the size of the spatial-domain FIR filter to apply to
  `x`. It is in the same units as `x`. The larger this is, the closer the output
  will be to the return value of `texshade.texshadeFFT`. This number controls
  the size of the neighborhood around a given pixel that contribute to that
  pixel's final texture-shaded value. If this is too small, the output will
  differ significantly from the full texture shading algorithm. If it is too
  big, you may run out of memory, because the overlap-save algorithm for
  fast-convolution will compute FFTs *at least* this size.

  **Spatial filter generation keyword args** passed to
  `texshade.hankel.halfHankel`: see that function's docstring for details, but
  reasonable values are chosen for these:

  - `interpMethod`
  - `sampleSpacing`
  - `hbTaps`
  - `hbTransitionWidth`

  **Overlap-save keyword args** passed to `ols.ols` (this function is in the
  `overlap-save` module on PyPI):

  - `size`
  - `nfft`
  - `out`

  `size`, a 2-list, specifies the size of the sub-arrays of the texture-shaded
  output to compute in each overlap-save step, while `nfft` (also a 2-list) is
  the size of the zero-padded FFT that will be taken at each overlap-save FFT.
  The requirement is that `nfft >= size + nDiameter - 1` for both dimensions. If
  `nfft` isn't provided, suitable numbers with small prime factors will be
  selected. If `size` isn't specified, a small multiple of `nDiameter` is
  chosen.
  
  N.B. It is beneficial to make `size` as big as can fit in your system memory.
  Suppose `nDiameter` is 1000. If you make `size=[15*1024, 15*1024]`,
  overlap-save will pick `nfft=[16*1024, 16*1024]` or a bit smaller. A 16k by
  16k array of float64 (actually, they'll be complex128, but the real-only FFT
  will only need half as much space, due to Fourier symmetry) uses 2 GB of
  memory. You'll probably need 4x this much to store all the intermediate
  FFT-related arrays:

  1. the FFT of the spatial filter,
  2. the FFT of the roughly 16k by 16k chunk of input
  3. the product of the two
  4. the inverse-FFT of the product

  I assume your input pixels are int16 or float32, so much smaller before FFT
  than after. So if your system has 8 GB free, you could pick `size=[15*1024,
  15*1024]`. A rough equation might be, if your system has `M` GB, let each
  element of `size` be roughly `np.sqrt(M / 4 * 1024**3 / 8) - nDiameter`.

  `out` allows you to specify the output array to store the results in. This is
  useful when you have a memory-mapped array prepared to accept the output of
  the algorithm, which will be float64. If `out.dtype` is not `float64`, then
  Numpy will perform a conversion, which might be expensive. If provided, this
  is returned. If not specified, a new array is allocated, filled, and returned.
  """
  h = halfHankel(
      nDiameter,
      alpha,
      interpMethod=interpMethod,
      sampleSpacing=sampleSpacing,
      hbTaps=hbTaps,
      hbTransitionWidth=hbTransitionWidth,
  )

  return ols(x, h, size=size, nfft=nfft, out=out)
```

```py
# export hankel-memmap.py
import numpy as np
from texshade import texshadeSpatial
import postprocess

fname = 'merged.tif.npy'
arr = np.load(fname, mmap_mode='r')

alpha = 0.8
Nwidth = 500
Nhalfband = 128

tex = np.lib.format.open_memmap('mmap-tex.npy', mode='w+', dtype=np.float64, shape=arr.shape)

texshadeSpatial(arr, alpha, Nwidth, hbTaps=Nhalfband, out=tex, size=[2000, 2000])

postprocess.texToPng(
    tex,
    'hankel-texshade-alpha-{}-n-{}-mmap.png'.format(alpha, Nwidth),
    quantiles=[.01, .99],
    borderFractions=[1e-2, 1e-2])
```

To downsample this large image for including with this repo:
```
convert -filter Mitchell -sampling-factor 1x1 -quality 90 -resize 2048 hankel-texshade-alpha-0.8-n-500-mmap.png hankel-texshade-alpha-0.8-n-500-mmap.png.small.png
```

![Memory-efficient tex-shading via memory-mapped files and overlap-save technique of fast convolution](hankel-texshade-alpha-0.8-n-500-mmap.png.small.png)

This image is qualitatively identical to the original texture-shaded output [previously shown](#tex-shaded) but to reemphasize: while the [original](#tex-shaded) texture-shaded image involved slurping the entire dataset into memory, and then computing huge FFTs of it, the image just above, generated by the `hankel-memmap.py` script, sipped memory: it loaded the input array as a memory-mapped file, allocated the output as a memory-mapped file, and used a spatial-domain filter to approximate the frequency-domain operator in the original image. By using the overlap-save method of fast-convolution, the filter was applied on small chunks of the data.

We can finally run the texture shading algorithm on enormous datasets without needing gargantuan amounts of memory.

## Developing in this repository

I edit README.md in any old text editor as a Knuth-style [literate program](https://en.wikipedia.org/wiki/Literate_programming), and run `$ node md2code.js` to
- tangle it into Python code, and running the [Yapf](https://github.com/google/yapf) Python code formatter on it, and
- gently weave the Markdown again with the Yapf-formatted code.

If you want to use [`md2code.js`](./mdcode.js) (which is totally uncommented and ad hoc), install [Node.js](https://nodejs.org) and run `$ npm i` in this repo.

To build the HTML, I use Pandoc and coordinate it with the [`Makefile`](./Makefile), which can be invoked by running `$ make`.

## Acknowledgements
John Otander's [Retro](http://markdowncss.github.io/retro/) CSS theme. KaTeX for rendering equations.