import numpy as np
import matplotlib.pyplot as plt
from pps import generate_plot, generate_plot_overlay, true_power_spectra

file_root_likelihood = "output_tightpriors/primordial_ps_likelihood_nDims8nLive100_polychord_raw/primordial_ps_likelihood_nDims8nLive100"
file_root_tightpriors = "output_tightpriors/primordial_ps_nDims8nLive100_polychord_raw/primordial_ps_nDims8nLive100"
file_root_fixedtau = "output_tightpriors_fixedtau/primordial_ps_nDims8nLive100_polychord_raw/primordial_ps_nDims8nLive100"

nDims = 8
title = "nDims = {}, Fixed tau".format(nDims)
xlim = [-4, -0.3]
ylim = [2,4]

# generate_plot(file_root_fixedtau, nDims, xlim, ylim, title = title)
# 

nDims = [8, 8]
file_roots = [file_root_fixedtau, file_root_tightpriors]

generate_plot_overlay(file_roots, nDims, xlim, ylim)