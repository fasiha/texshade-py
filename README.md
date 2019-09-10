# Texshade: texture-shaded elevation via the fractional-Laplacian operator

## Introduction

See http://www.textureshading.com/Home.html for links to papers and slides by Leland Brown from 2010 and 2014 describing the technique of texture shading, but in summary, it is a method of processing digital elevation maps (DEMs) that highlights the network nature of topography, throwing ridges, canyons, and valleys into sharp relief.

This repository contains an open-source public-domain Python/Numpy software library to apply the texture shading algorithm on *extremely* large datasets. This is a challenge because a straightforward implementation of the texture-shading technique requires loading the entire elevation map into memory. For large datasets—like the ASTER Global DEM, which comes in at roughly 250 GB compressed—you either have to find a computer with a lot of memory (nodes with 1 TB RAM are available at many scientific organizations as of 2018) or you have to modify the technique.

This repository contains (1) mathematical and (2) software details of a low-memory approximation to the original texture-shading algorithm that in practice produces texture-shaded imagery very similar to the full algorithm.

The mathematical trick, in a nutshell, is to use the Hankel transform to find a finite impulse response (FIR) filter that approximates the frequency-domain fractional-Laplacian operator, and apply that filter in the spatial domain via the efficient overlap-save algorithm. According to GitHub commit logs, I first derived this technique in 2015.

## The texture-shading algorithm

The original texture-shading algorithm takes a 2D array of elevations, call it \\(x\\), and computes the texture-shaded elevation map,

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

nextpow2 = lambda v: list(map(int, 2**np.ceil(np.log2(v))))


def texshade(x, alpha, verbose=True):
  Nyx = nextpow2(x.shape)

  fy = scifft.rfftfreq(Nyx[0])[:, np.newaxis].astype(x.dtype)
  fx = scifft.rfftfreq(Nyx[1])[np.newaxis, :].astype(x.dtype)
  H2 = (fx**2 + fy**2)**(alpha / 2.0)
  if verbose:
    print("Generated filter")

  rfft2 = lambda x: scifft.rfft(scifft.rfft(x, Nyx[1], 1, True), Nyx[0], 0, True)
  irfft2 = lambda X: scifft.irfft(scifft.irfft(X, axis=0, overwrite_x=True), overwrite_x=True)

  xr = rfft2(x) * H2
  if verbose:
    print("Completed frequency domain operations")
  H2 = None  # potentially trigger GC here to reclaim H2's memory
  xr = irfft2(xr)
  if verbose:
    print("Back to spatial-domain")

  return xr[:x.shape[0], :x.shape[1]]
```

## Test setup
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

```py
# export convert.py
"""
Quick script intended to be used only by a user to convert a specific
GeoTIF to a NPY file for pure-Numpy non-GDAL demo.
"""
import numpy as np
import gdal, gdalconst
fname = 'merged.tif'


def filenameToData(fname, dtype=np.float32):
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

```py
# export postprocess.py
import numpy as np


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
```

```
for i in orig.png scaled.png; do convert -filter Mitchell -sampling-factor 1x1 -quality 90 -resize 2048 \\(i \\)i.small.png; done
```

### Original
![original downsampled](orig.png.small.png)

### Tex-shaded
![tex-shaded downsampled](scaled.png.small.png)

## The approximation

Is there any way to apply the fractional-Laplacian operator, which is expressed in the frequency-domain as \\(|\vec f|^α ⋅ F[x]\\) for an input array \\(x\\), that *doesn't* require a 2D Fourier transform of \\(x\\)? Recall that the Fourier transform is a unitary operator—that is, \\(F[x]\\) can be seen as a matrix–vector product \\(\underline F ⋅ \underline x\\), where the underlines represent a matrix or vector version of the operator or its input, and \\(\underline F\\) is a unitary matrix (the complex-domain extension of an orthogonal matrix). This indicates that each element of the output of a Fourier transform is a function of each of the input elements (though because it can do this in \\(log(N)\\) operations, instead of \\(N\\), we call it the *fast* Fourier transform (FFT)). There doesn't seem to be a memory-local way to convert the array of elevations to the Fourier domain, since each frequency bin has contributions from each pixel in the elevation array.

But we do know from linear systems theory that multiplication in the frequency domain is equivalent to convolution in the spatial domain. We can ask if there's any structure to the spatial-domain representation of the fractional-Laplacian \\(|\vec f|^α\\), i.e., what is \\(F^{-1}[|\vec f|^α]\\)?

Trawling through the Wikipedia I stumbled on [the Hankel transform and its relationship to the Fourier transform of circularly-symmetric functions](https://en.wikipedia.org/w/index.php?title=Hankel_transform&oldid=901300195#Relation_to_the_Fourier_transform_(circularly_symmetric_case)). Wikipedia notes that for a two-dimensional radial function \\(f(r)\\), its two-dimensional Fourier transform \\(F(\vec k)\\) is

$$F(\vec k) = F(k) = 2π\int_0^{\infty} f(r) ⋅ J_0(k r) \cdot r ⋅ dr,$$
where \\(J_0(⋅)\\) is the Bessel function of the first kind of order 0. In our notation, if we represent the fractional-Laplacian operator as \\(l(\vec f) = l(f) = f^α\\) ("l" for "Laplacian"), its Fourier transform is, according to [Wolfram Alpha](https://www.wolframalpha.com/input/?i=2*pi*Integrate%5Bf%5Ea+*+BesselJ%5B0%2C+k+*+f%5D+*+f%2C+f%2C+0%2C+m%5D),

$$2π\int_0^m f ⋅ f^α J_0(f r) df = 2π \frac{m^{α + 2}}{α + 2} \cdot {}_{1}F_2([α / 2 + 1], [1, α / 2 + 2], -(r ⋅ m / 2)^2)$$
where
- \\(r\\) is my variable for the radius in the spatial domain,
- \\(1F2\\) is a generalized hypergeometric function (not "the" hypergeometric function \\(2F1\\)!), and
- where I left the upper limit of the integral as \\(m\\) (for "max") because we have a bound on the extent of the frequency domain \\(\vec f = [f_x, f_y]'\\), since \\(-π ≤ f_x < π\\) radians per pixel, and same for \\(f_y\\). (Recall this happens because we are working with a discrete-valued array of elevations \\(x\\), so the Fourier transform is a discrete-time Fourier transform (DTFT) and is periodic every 2π radians per pixel.)

> Odd sidebar. My little knowledge of mathematics is exhausted by wondering why, if I omit the 2π in the expression to Wolfram Alpha, it returns a much more complicated expression including Γ functions. Sympy similar story.

The constant factors that accrete when working with Fourier transform pairs are usually incredibly tedious to keep track of, especially when evaluating them with the FFT. My normal practice is to get things working up to a constant factor and then see if I need to worry about that factor.

So let us ask what the Fourier transform of an array containing evaluations of the radial function

$$L(r) = {}_{1}F_2([α / 2 + 1], [1, α / 2 + 2], -(r ⋅ π / 2)^2).$$
We use the maximum of the integral in the Hankel transform is \\(m=π\\). Recall we use "l" and "L" for "Laplacian": \\(L(r)\\) is the Fourier transform of the fractional-Laplacian \\(l(f) = |f|^α\\).

We do this in the code snippet below: we evaluate the above generalized hypergeometric function on a 200×200 array of radii. We assume the array's horizontal and vertical axes run from -100 to 99, i.e., assuming one pixel spacing for each element, compute each element's radius \\(r\\), and evaluate \\(L(r)\\). Then we look at its 2D FFT, which will be all-real because the input is symmetric. (Recall that in general, the Fourier transform of a real vector will contain complex entries but be conjugate-symmetric about the origin. The Fourier transform will contain zero imaginary components only if its input was symmetric about the origin.)

```py
# export math-hankel.py
import numpy as np
from mpmath import hyper
import numpy.fft as fft
import pylab as plt
plt.style.use('ggplot')


def spatial(r, a):
  "Evaluate L(r), proportional to the Fourier transform of |f|**α"
  return float(hyper((0.5 * a + 1,), (1.0, 0.5 * a + 2), -0.25 * (r * np.pi)**2))


xmat, ymat = np.meshgrid(np.arange(-100, 100), np.arange(-100, 100))
rmat = np.sqrt(xmat**2 + ymat**2)
alpha = 0.8
h = np.vectorize(lambda r: spatial(r, alpha))(rmat)
```

Above we use the fabulous [`mpmath`](http://mpmath.org/) package—a pure-Python arbitrary-precision package with extensive support for special functions, quadrature integration, linear algebra, etc., started by Fredrik Johansson in 2007 (when he was a teenager)—to compute the generalized hypergeometric function. Next, we'd like to visualize its Fourier transform—hopefully we see something that looks like \\(|f|^{0.8}\\).

```py
# export math-hankel.py
def F2cent(arr):
  """Origin-centered 2D Fourier transform"""
  return fft.fftshift(fft.fft2(fft.ifftshift(arr)))


def plotF2cent(arr):
  """Given an origin-centered 2D array, plot its 2D Fourier transform"""

  def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta / 2, f[-1] + delta / 2]

  h, w = arr.shape
  x = np.ceil(np.arange(w) - w / 2) / w
  y = np.ceil(np.arange(h) - h / 2) / h
  fig, (sax, fax) = plt.subplots(1, 2)

  sax.imshow(
      arr,
      aspect='equal',
      interpolation='none',
      extent=extents(x * w) + extents(y * h),
      origin='lower')

  fax.imshow(
      np.real(F2cent(arr)),
      aspect='equal',
      interpolation='none',
      extent=extents(x) + extents(y),
      origin='lower')
  sax.grid(False)
  fax.grid(False)
  sax.set_xlabel('pixel')
  fax.set_xlabel('cycles/pixel')

  return fig, sax, fax


hplots = plotF2cent(h)
hplots[1].set_title('L(r): spatial-domain')
hplots[2].set_title('F[L(r)]: frequency-domain')
plt.savefig('full-hankel.png', dpi=300, bbox_inches='tight')
plt.savefig('full-hankel.svg', bbox_inches='tight')

actual = np.real(F2cent(h))[100, :]
f = np.ceil(np.arange(200) - 200 / 2) / 200
expected = np.abs(f * 4)**alpha
plt.figure()
plt.plot(f, actual, '-', f, expected, '--', f, actual / expected, '.')
plt.xlabel('f (cycles/pixel)')
plt.legend(['actual', 'expected', 'actual/expected'])
plt.title('Cut of actual F[L(r)] versus expected |4⋅f|^0.8')
plt.savefig('full-hankel-actual-expected.png', dpi=300, bbox_inches='tight')
plt.savefig('full-hankel-actual-expected.svg', bbox_inches='tight')
```

![An array of evaluating the expression we computed for the Fourier transform of the fractional-Laplacian operator, and it's actual Fourier transform](full-hankel.png)

Above: an array of evaluating the expression we computed for the Fourier transform of the fractional-Laplacian operator, and it's actual Fourier transform.

![Comparing the actual frequency response of our expression for the spatial-domain equivalent of the fractional-Laplacial, versus the expected frequency response, and their difference, which is approximately 1.024](full-hankel-actual-expected.svg)

Above: Comparing the actual frequency response of our expression for the spatial-domain equivalent of the fractional-Laplacial, versus the expected frequency response, and their difference, which is approximately 1.024.

This is a success! First, note that the 200×200 array on the left is close to zero: it has a bright center, and decays quickly as the radius from the center–origin grows. Next, note that its 2D Fourier transform is indeed what we had hoped: it sweeps out \\(∝|f|^{α=0.8}\\) radially, for angular radii between 0 and π radians (normalized here to cycles instead of radians: the axes between ±0.5 cycles per pixel correspond to ±π radians per pixel). (The symbol "\\(∝\\)" is read as "proportional to".)

The second plot above shows the near-constant ratio between the center-cut through the FFT of the spatial filter \\(L(r)\\) and a scaled version of what we expect. Comparing \\(|4f|^α\\), for \\(α=0.8\\), to the cut through the FFT's output, we see a very-nearly-constant ratio of 1.024. Do note that the actual value inside the absolute value is irrelevant, and amounts only to scaling the texture-shaded output.

**However**, we cannot use \\(L(r)\\) as a spatial-domain equivalent of texture-shading because recall that the original algorithm requires

$$y = F^{-1}[F[x] ⋅ |\vec f|^α],$$
but \\(|\vec f|^α\\) includes the *corners* of the frequency domain, not the radial pattern we see from the circular bull's-eye chart above, where the corners in the frequency domain get zero weight. We might use \\(L(r)\\) nonetheless and accept the infidelity to the texture-shading algorithm, but we don't need to. If we decimate the spatial-domain filter \\(L(r)\\) by two, then we effectively get the middle-half of its frequency response, which will be \\(∝|f|^α\\) all the way out to its edges. The Scipy ecosystem provides several ways to [design halfband filters](https://docs.scipy.org/doc/scipy/reference/signal.html#filter-design). A simple example to demonstrate the idea will suffice: design an 8th order low-pass Butterworth filter and apply it along rows and columns of the spatial-domain filter, then downsample the result (throw away every other row/column):

```py
# export math-hankel.py
import scipy.signal as sig
lpf = sig.iirfilter(8, 0.5, btype='lowpass', ftype='butter')
hiir = sig.filtfilt(*lpf, sig.filtfilt(*lpf, h, axis=0), axis=1)

lpfplots = plotF2cent(hiir)
decplots = plotF2cent(hiir[:-1:2, :-1:2])

lpfplots[1].set_title('LPF[L(r)]')
lpfplots[2].set_title('F[LPF[L(r)]]')
lpfplots[0].savefig('lpf-hankel.png', dpi=300, bbox_inches='tight')
lpfplots[0].savefig('lpf-hankel.svg', bbox_inches='tight')

decplots[1].set_title('HB[L(r)]')
decplots[2].set_title('F[HB[L(r)]]')
decplots[0].savefig('hb-hankel.png', dpi=300, bbox_inches='tight')
decplots[0].savefig('hb-hankel.svg', bbox_inches='tight')
```

The results are positive: while this filtering process can certainly be improved, we have obtained a spatial-domain filter that closely-approximates the fractional-Laplacian frequency-domain operator needed by the texture-shading algorithm.

![Low-pass-filtered version of our initial spatial-domain-created filter](lpf-hankel.png)

Above: low-pass-filtered version of our initial spatial-domain-created filter.

![Halfbanded (decimated) version of our initial spatial-dmain-created filter: this meets the requirements of the original texture-shading algorithm](hb-hankel.png)

Above: halfbanded (decimated) version of our initial spatial-dmain-created filter: this meets the requirements of the original texture-shading algorithm.

## A more complete implementation of the approximation

```py
# export texshade/hankel.py
from mpmath import hyper
import numpy as np
from scipy import signal
from scipy.signal import convolve2d
import numpy.fft as fft
import functools
from scipy.interpolate import interp1d


@functools.lru_cache(maxsize=None)
def spatial(r, a, integralMax=np.pi):
  # Wolfram Alpha: `2*pi*Integrate[f^a * BesselJ[0, k * f] * f, f, 0, m]`
  return float(hyper((0.5 * a + 1,), (1.0, 0.5 * a + 2), -0.25 * (r * integralMax)**2))


def vec(v):
  return v.reshape(v.size, -1)


def fullHankel(n, alpha, interpMethod=True, sampleSpacing=None):
  rvec = np.arange(-n, n)
  rmat = np.sqrt(vec(rvec)**2 + vec(rvec).T**2)

  if interpMethod:
    if not sampleSpacing:
      sampleSpacing = 1e-2
    data = precomputeLoad(alpha, np.ceil(np.sqrt(2) * n * 1.01), sampleSpacing)
    oned = interp1d(data['r'], data['h'])
    hmat = oned(rmat)
  else:
    fun = np.vectorize(lambda x: spatial(x, alpha))
    hmat = fun(rmat)
  return hmat


def designHalfband(N, passbandWidth):
  if N % 2 != 0:
    raise ValueError('N must be even')
  if N < 2:
    raise ValueError('N must be > 1')
  if not (passbandWidth > 0 and passbandWidth < 0.5):
    raise ValueError('Need 0 < passbandWidth < 0.5')
  bands = np.array([0., .25 - passbandWidth, .25 + passbandWidth, .5])
  h = signal.remez(N + 1, bands, [1, 0], [1, 1])
  h[abs(h) <= 1e-4] = 0.0
  return h


def halfband(hmat, taps=128, passbandWidth=0.03):
  hbFilter = designHalfband(taps, passbandWidth)
  doubleFilter = convolve2d(
      convolve2d(hmat, vec(hbFilter), mode='same'), vec(hbFilter).T, mode='same')
  n = hmat.shape[0]
  finalFilter = doubleFilter[:-1:2, :-1:2] if n % 4 == 0 else doubleFilter[1:-1:2, 1:-1:2]
  return finalFilter


def halfHankel(nDiameter,
               alpha,
               interpMethod=True,
               sampleSpacing=None,
               hbTaps=128,
               hbPassbandWidth=0.03):
  return halfband(
      fullHankel(nDiameter, alpha, interpMethod=interpMethod, sampleSpacing=sampleSpacing),
      taps=hbTaps,
      passbandWidth=hbPassbandWidth)


def precomputeLoad(alpha, N, spacing):
  import os.path
  fun = np.vectorize(lambda x: spatial(x, alpha))
  r = np.arange(0, N, spacing)
  fname = 'hankel-alpha-{}'.format(alpha)
  if os.path.isfile(fname + '.npz'):
    npz = np.load(fname + '.npz')
    rsave = npz['x']
    hsave = npz['y']

    rnew = np.sort(np.setdiff1d(r, rsave))
    hnew = fun(rnew) if rnew.size > 0 else []

    r = np.hstack([rsave, rnew])
    h = np.hstack([hsave, hnew])
    idx = np.argsort(r)  # wasteful but resorting 1k vec is fast
    r = r[idx]
    h = h[idx]
  else:
    h = fun(r)
  np.savez(fname, x=r, y=h)
  return dict(r=r, h=h)
```

```py
# export hankel-demo.py
import numpy as np
import texshade
import postprocess
from scipy.signal import fftconvolve
nextpow2 = lambda v: list(map(int, 2**np.ceil(np.log2(v))))

fname = 'merged.tif.npy'
arr = np.load(fname)

clip = True
if clip:
  arr = arr[-1500:, -1500:]


def texToFile(tex, fname):
  minmax = np.quantile(tex.ravel(), [.01, .99])
  scaled = postprocess.touint(tex, minmax[0], minmax[1], np.uint8)
  postprocess.toPng(scaled, fname)


alpha = 0.8

texToFile(
    texshade.texshadeFFT(arr, alpha),
    'orig-texshade-alpha-{}{}.png'.format(alpha, '-clip' if clip else ''))

Nwidth = 500
Nhalfband = 128

h = texshade.halfHankel(Nwidth, alpha, hbTaps=Nhalfband)
print('halfbanded', h.shape)
texToFile(
    fftconvolve(arr, h, mode='same'),
    'hankel-texshade-alpha-{}-n-{}{}.png'.format(alpha, Nwidth, '-clip' if clip else ''))

hFull = texshade.hankel.fullHankel(Nwidth, alpha)
print('non-halfbanded', hFull.shape)
texToFile(
    fftconvolve(arr, hFull, mode='same'),
    'hankel-texshadeFullband-alpha-{}-n-{}{}.png'.format(alpha, Nwidth, '-clip' if clip else ''))
```

We can compare the output of the original texture-shading algorithm:

![Texshaded clip](orig-texshade-alpha-0.8-clip.png)

with that of the Hankel approximation:

![Hankel-approximated texshaded clip](hankel-texshade-alpha-0.8-n-500-clip.png)

Both are very close. Toggling between them only reveals slight contrast differences due to the different levels obtained for the quantiles—these differences are likely caused by the artifacts at the edges.

In the above demo code, we ask Scipy's [`fftconvolve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html) to apply the spatial-domain filtering, which behind the scenes uses a full-sized FFT just like the original method. However, we can make this much more memory-efficient, while retaining its speed, by using the overlap-add (or overlap-save) technique of fast-convolution.

## Overlap-save method for fast-convolution
The textbook definition of convolving two signals in the spatial domain is a quadratic \\(O(N^2)\\) operation. Since convolution in the spatial domain is mathematically equivalent to multiplication in the frequency domain, and the FFT is a log-linear \\(O(N \log N)\\) operation, this is usually much faster—this is why we use `fftconvolve` above. The drawback of the FFT-based alternative to direct convolution is that it requires we run the FFT on the signals of interest—with a potentially prohibitive memory burden.

The overlap-save method (and its closely-related sibling, the overlap-add method) allow us to convolve signals more intelligently: it still uses FFTs, so the overall theoretical runtime complexity remains log-linear, but it uses *many small* FFTs so memory consumption remains reasonable. I prefer overlap-save because it partitions the *output* array into non-overlapping segments that each step of the algorithm fills in (and which may be parallelized). Each step of the overlap-save algorithm reaches for segments of *input* that may overlap with other steps, but this overlap is read-only.

> In contrast, overlap-add splits the *input* array into non-overlapping segments. Each step of that algorithm has to potentially modify previously-computed samples of the *output*, which makes parallelization much more nasty (requiring locks or careful orchestration of the sequence of steps).

The overlap-save implementation I wrote is largely out of the scope of this texture-shading library, so let's just import it and show how we can use it, along with memory-mapped inputs and outputs to *really* save memory.

```py
# export hankel-memmap.py
import numpy as np
from texshade import halfHankel
import postprocess
from ols import ols

fname = 'merged.tif.npy'
arr = np.load(fname, mmap_mode='r')


def texToFile(tex, fname):
  minmax = np.quantile(tex.ravel(), [.01, .99])
  scaled = postprocess.touint(tex, minmax[0], minmax[1], np.uint8)
  postprocess.toPng(scaled, fname)


alpha = 0.8

Nwidth = 500
Nhalfband = 128
h = halfHankel(Nwidth, alpha, hbTaps=Nhalfband)

tex = np.lib.format.open_memmap('mmap-tex.npy', mode='w+', dtype=np.float64, shape=arr.shape)
ols(arr, h, size=[2000, 2000], out=tex)
texToFile(tex, 'hankel-texshade-alpha-{}-n-{}-mmap.png'.format(alpha, Nwidth))
```

To downsample this large image for including with this repo:
```
convert -filter Mitchell -sampling-factor 1x1 -quality 90 -resize 2048 hankel-texshade-alpha-0.8-n-500-mmap.png hankel-texshade-alpha-0.8-n-500-mmap.png.small.png
```

![Memory-efficient tex-shading via memory-mapped files and overlap-save technique of fast convolution](hankel-texshade-alpha-0.8-n-500-mmap.png.small.png)

This image is qualitatively identical to the original texture-shaded output [previously shown](#tex-shaded) but to reemphasize: while the [original](#tex-shaded) texture-shaded image involved slurping the entire dataset into memory, and then computing huge FFTs of it, the image just above, generated by the `hankel-memmap.py` script, sipped memory: it loaded the input array as a memory-mapped file, allocated the output as a memory-mapped file, and used a spatial-domain filter to approximate the frequency-domain operator in the original image. By using the overlap-save method of fast-convolution, the filter was applied on small chunks of the data.

We can finally run the texture-shading algorithm on enormous datasets without needing gargantuan amounts of memory.

## Acknowledgements
John Otander's [Retro](http://markdowncss.github.io/retro/) CSS theme. KaTeX for rendering equations.
