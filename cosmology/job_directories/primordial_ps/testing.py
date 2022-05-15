from http.client import NON_AUTHORITATIVE_INFORMATION
import numpy as np
import matplotlib.pyplot as plt

from pps.files import run_exists, return_valid_internal_points
from pps.plotting import return_evidence, generate_file_root, extend_samples, manipulate_axes
from pps.theory import plf, get_params_from_nDims
from pps.priors import SortedUniformPrior, UniformPrior, hypercube_to_theta, get_prior_samples, get_prior_weights

from fgivenx import plot_dkl, samples_from_getdist_chains, plot_contours
from fgivenx.drivers import compute_dkl

def generate_posterior_comparison(nInternalPoints_list, fixed, f_xlim, ylim, a_xlim, nX = 200, nY = 500, title = None, axs = None, plot_xlabel = True, plot_ylabel = True, plot_xlabel_top = True):
    """
        Generates a posterior plot marginalised over all parameters. 
    """

    N = len(nInternalPoints_list)

    nDims_list = nInternalPoints_list * 2 + 2
    
    samples1 = [None] * N
    weights1 = [None] * N
    samples2 = [None] * N
    weights2 = [None] * N
    logZ1 = [None] * N
    logZ2 = [None] * N
    cache1 = [None] * N 
    cache2 = [None] * N

    for i in range(N):
        nInternalPoints = nInternalPoints_list[i]
        nDims = nDims_list[i]

        params_list, _ = get_params_from_nDims(nDims)

        # Yes I know this is bad programming but so what. 
        file_root1 = generate_file_root(nInternalPoints, fixed[0])
        sample1, weight1 = samples_from_getdist_chains(params_list, file_root1)
        cache1[i] = generate_file_root(nInternalPoints, fixed[0], cache = True) + 'comparison1'
        samples1[i] = sample1 
        weights1[i] = weight1
        logZ1[i] = return_evidence(file_root1)[0]

        file_root2 = generate_file_root(nInternalPoints, fixed[1])
        sample2, weight2 = samples_from_getdist_chains(params_list, file_root2)
        cache2[i] = generate_file_root(nInternalPoints, fixed[1], cache = True) + 'comparison1'
        samples2[i] = sample2
        weights2[i] = weight2
        logZ2[i] = return_evidence(file_root2)[0]

    samples1 = extend_samples(samples1)
    samples2 = extend_samples(samples2)

    x = np.linspace(a_xlim[0], a_xlim[1], nX)

    x_prior = SortedUniformPrior(f_xlim[0], f_xlim[1])
    y_prior = UniformPrior(ylim[0], ylim[1])

    def plf_adjusted(x, hypercube):
        theta = hypercube_to_theta(hypercube, x_prior, y_prior)
        return plf(x, theta, f_xlim)

    f = [plf_adjusted] * N

    if not axs:
        fig, axs = plt.subplots()
    
    cbar1 = plot_contours(f, x, samples1, axs, logZ = logZ1, weights = weights1, cache = cache1[0], colors=plt.cm.Blues_r, lines = False, ny = nY)
    
    cbar2 = plot_contours(f, x, samples2, axs, logZ = logZ2, weights = weights2, cache = cache2[0], ny = nY)
    
    axs.set_ylim(ylim)

    if title:
        axs.set_title(title)

    print(x)
    return cbar1, cbar2

fig, axs = plt.subplots()
nInternalPoints = np.array([0,1,2,3,4,5,6,7])
fixed = [None, 'tau']
xlim = [-4, -0.3]
ylim = [2, 4]
new_xlim = [-2, -1]
new_ylim = [2.95, 3.15]

cbar1, cbar2 = generate_posterior_comparison(nInternalPoints_list = nInternalPoints, fixed = fixed, f_xlim = xlim, ylim = ylim, a_xlim = new_xlim, axs = axs)



axs.set_ylim(new_ylim)
axs.set_xlim(new_xlim)

fig.tight_layout(rect = (0.05,0.05,0.95, 0.95))
cbar2 = fig.colorbar(cbar2, ax = axs, fraction = 0.03, aspect = 50, pad = 0.01)
cbar2.ax.tick_params(axis = 'y', which = 'both', direction = 'in')
cbar2.ax.set_title(r'$\sigma_{\tau}$')

cbar1 = fig.colorbar(cbar1, ax = axs, fraction = 0.03, aspect = 50, pad = 0.03)
cbar1.ax.set_yticks([])
cbar1.ax.set_title(r'$\sigma_f$')

axs = manipulate_axes(axs, dkl = False, plot_xlabel = True, plot_ylabel = True, plot_xlabel_top = True, xlim = new_xlim)



plt.savefig('output.png')
plt.savefig('output.eps', format = 'eps')