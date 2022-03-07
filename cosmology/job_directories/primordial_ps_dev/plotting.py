import numpy as np
import matplotlib.pyplot as plt
from PowerSpectraModule import generate_plot

file_root = "output/primordial_ps_nDims6nLive100_polychord_raw/primordial_ps_nDims6nLive100"
nDims = 6
xlim = [-4, -0.3]
ylim = [2, 4]

generate_plot(file_root, nDims, xlim, ylim)