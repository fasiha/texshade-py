# -*- coding: utf-8 -*-

import scipy.fftpack as scifft
import numpy as np

nextpow2 = lambda v: 1 << int(np.ceil(np.log2(v)))


def texshade(x, alpha, verbose=True):
    Nyx = nextpow2(x.shape)

    fy = scifft.rfftfreq(Nyx[0])[:, np.newaxis].astype(x.dtype)
    fx = scifft.rfftfreq(Nyx[1])[np.newaxis, :].astype(x.dtype)
    H2 = (fx**2 + fy**2)**(alpha / 2.0)
    if verbose: print "Generated filter"

    rfft2 = lambda x: scifft.rfft(scifft.rfft(x, Nyx[1], 1, True), Nyx[0], 0,
                                  True)
    irfft2 = lambda X: scifft.irfft(scifft.irfft(X, axis=0, overwrite_x=True),
                                    overwrite_x=True)

    xr = rfft2(x) * H2
    if verbose: print "Completed frequency domain operations"
    H2 = None  # potentially trigger GC here to reclaim H2's memory
    xr = irfft2(xr)
    if verbose: print "Back to spatial-domain"

    return xr[:x.shape[0], :x.shape[1]]
