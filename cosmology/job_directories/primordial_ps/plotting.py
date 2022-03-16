import numpy as np
import matplotlib.pyplot as plt
from pps.plotting import generate_plot

file_root_likelihood = "output_tightpriors/primordial_ps_likelihood_nDims8nLive100_polychord_raw/primordial_ps_likelihood_nDims8nLive100"
file_root_tightpriors = "output_tightpriors/primordial_ps_nDims8nLive100_polychord_raw/primordial_ps_nDims8nLive100"
file_root_fixedtau = "output_tightpriors_fixedtau/primordial_ps_nDims8nLive100_polychord_raw/primordial_ps_nDims8nLive100"

file_root = "output/primordial_ps_nInternalPoints2nLive100_polychord_raw/primordial_ps_nInternalPoints2nLive100"
file_root = "output/primordial_ps_nInternalPoints4nLive100_polychord_raw/primordial_ps_nInternalPoints4nLive100"
file_root = "output/primordial_ps_nInternalPoints6nLive100_polychord_raw/primordial_ps_nInternalPoints6nLive100"
file_root = "output/primordial_ps_nInternalPoints8nLive100_polychord_raw/primordial_ps_nInternalPoints8nLive100"


nInternalPoints = 8
nDims = nInternalPoints * 2 + 2
title = "nInternalPoints = {}".format(nInternalPoints)
xlim = [-4, -0.3]
ylim = [2,4]

generate_plot(file_root, nDims, xlim, ylim, title = title)