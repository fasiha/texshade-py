from setuptools import setup

setup(
    name='texshade',
    version='2.0',
    author='Ahmed Fasih',
    author_email='ahmed@aldebrn.me',
    description="Memory-efficient implementation of texture-shading method for visualizing elevation",
    long_description="Texture-shading via a spatial-domain transform of the full Fourier-domain fractional-Laplacian via the Hankel transform and overlap-save method of fast-convolution",
    license='Unlicense',
    url='https://github.com/fasiha/texshade-py',
    packages=['texshade'],
    install_requires='array-range,nextprod,numpy,overlap-save,scipy'.split(','),
    zip_safe=True,
    keywords='texture shading elevation terrain fractional laplacian hankel overlap save spatial frequency',
)
