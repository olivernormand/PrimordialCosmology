import numpy as np
import matplotlib.pyplot as plt
from lineorsin.plotting import generate_plot

file_root = "chains/model_comparison_polychord_raw/model_comparison"
file_root = "chains/model_comparison_18_polychord_raw/model_comparison_18"

nDims = 18
title = "nDims = {}".format(nDims)
xlim = [0, 1]
ylim = [-2,2]

generate_plot(file_root, nDims, xlim, ylim, title = title)