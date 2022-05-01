import numpy as np
import matplotlib.pyplot as plt
from pps.plotting import generate_plot, generate_plots, generate_dkl_plot, generate_file_root

nInternalPoints = np.array([3])
fixed = [True]
xlim = [-4, -0.3]
ylim = [2,4]

generate_plots(nInternalPoints, fixed, xlim, ylim)



# file_roots = [fr1, fr2, fr3, fr4]
# nInternalPoints = np.array([2, 4, 6, 8])
# nDims = nInternalPoints * 2 + 2

# generate_dkl_plot(file_roots, nDims, xlim, ylim, 'Dkl Plot')