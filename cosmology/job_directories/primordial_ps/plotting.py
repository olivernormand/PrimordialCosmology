import numpy as np
import matplotlib.pyplot as plt
from PowerSpectraModule import generate_plot

#file_root = "output/primordial_ps_nDims6nLive100_polychord_raw/primordial_ps_nDims6nLive100"
file_root = "output_100nLive_nDims2/primordial_ps_polychord_raw/primordial_ps"
file_root = "output/primordial_ps_polychord_raw/primordial_ps"
nDims = 6
xlim = [-4, -0.3]

generate_plot(file_root, nDims, xlim)