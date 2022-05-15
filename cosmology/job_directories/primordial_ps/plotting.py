import numpy as np
import matplotlib.pyplot as plt
from pps.plotting import generate_single_posterior_plot, generate_plots_array, generate_dkl_plot, bayes_factor, generate_posterior_plot, generate_dkl, generate_marginalised_posterior_array, generate_dkl_new
from pps.files import generate_file_root

nInternalPoints = np.array([0,1,2,3,4,5,6,7])
parameters = [None, 'all', 'H0', 'ombh2', 'omch2', 'tau']
fixed = None
xlim = [-4,-0.3]
ylim = [2,4]



generate_plots_array(nInternalPoints, fixed, xlim, ylim)
#bayes_factor(rds = True)
#generate_dkl_plot(fixed = parameters)
#generate_posterior_plot(nInternalPoints, parameters[0], xlim, ylim)
#generate_dkl(nInternalPoints, parameters[0], xlim, ylim)
generate_marginalised_posterior_array()
#
# nInternalPoints = 1
# nDims = nInternalPoints * 2 + 2
# file_root = generate_file_root(nInternalPoints = 1, fixed = None, cache = False, rds = False)
# # print(file_root)
# generate_single_posterior_plot(file_root, nDims, xlim, ylim, fig_title = 'Redoing plot', fig_name = 'output.png', nY = 100)

# dkls = np.zeros([6, 100])
# for i in range(6):
#     dkls[i,:] = generate_dkl_new(fixed = parameters[i])
# dkl = generate_dkl_new(fixed = parameters[i])
# plt.plot(dkl)
# plt.savefig('output{}.png'.format(i))