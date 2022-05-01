import matplotlib.pyplot as plt
import numpy as np
from fgivenx import plot_contours, plot_dkl, samples_from_getdist_chains

from pps.theory import get_params_from_nDims, plf
from pps.priors import UniformPrior, SortedUniformPrior, hypercube_to_theta


def generate_plot(file_root, nDims, xlim, ylim, fig_title=None, fig_name = None, plot_function=None, axs = None):
    params_list, _ = get_params_from_nDims(nDims)

    samples, weights = samples_from_getdist_chains(params_list, file_root)

    x = np.linspace(xlim[0], xlim[1], 200)

    x_prior = SortedUniformPrior(xlim[0], xlim[1])
    y_prior = UniformPrior(ylim[0], ylim[1])

    def plf_adjusted(x, hypercube):
        theta = hypercube_to_theta(hypercube, x_prior, y_prior)
        return plf(x, theta, xlim)

    if not axs:
        fig, axs = plt.subplots()
    cbar = plot_contours(plf_adjusted, x, samples, axs, weights=weights, ny = 500)
    cbar = plt.colorbar(cbar, ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(['', r'$1\sigma$', r'$2\sigma$', r'$3\sigma$'])

    par1 = axs.twiny()
    axs.tick_params(axis = 'x', colors = 'white')

    par1.set_xlim(10**(-4), 10**(-0.3))
    par1.set_xscale('log')
    par1.xaxis.set_ticks_position('bottom')

    if fig_title:
        axs.set_title(fig_title)
    axs.set_ylim(ylim)
    axs.set_ylabel(r"$ \ln (10^{10} \mathcal{P}_{\mathcal{R}})$")
    axs.set_xlabel(r"$ k \quad $ [Mpc]$^{-1} $", loc = 'bottom')

    if plot_function:
        axs.plot(x, plot_function(x), 'g')

    fig.tight_layout()
    if not axs:
        if fig_name:
            plt.savefig(fig_name)
        else:
            plt.savefig('output.png')

def generate_plots(nInternalPoints_list, fixed_list, xlim, ylim, single_fig = False):

    for nInternalPoints in nInternalPoints_list:
        
        nDims = nInternalPoints * 2 + 2
        
        for fixed in fixed_list:
            file_root = generate_file_root(nInternalPoints, fixed)
            title = "nInternalPoints = {}".format(nInternalPoints)
            name = "output_figures/nInt{}fixed{}".format(nInternalPoints, int(fixed))
            generate_plot(file_root, nDims, xlim, ylim, fig_title = title, fig_name = name)


def generate_dkl_plot(file_roots, nDims_list, xlim, ylim, title = None):
    N = len(file_roots)
    
    samples = [None] * N
    weights = [None] * N
    prior_samples = None

    for i in range(N):
        nDims = nDims_list[i]
        params_list, _ = get_params_from_nDims(nDims)
        file_root = file_roots[i]
        sample, weight = samples_from_getdist_chains(params_list, file_root)
        samples[i] = sample 
        weights[i] = weight

    for sample in samples:
        print(sample.shape)

    x = np.linspace(xlim[0], xlim[1], 200)

    x_prior = SortedUniformPrior(xlim[0], xlim[1])
    y_prior = UniformPrior(ylim[0], ylim[1])

    def plf_adjusted(x, hypercube):
        theta = hypercube_to_theta(hypercube, x_prior, y_prior)
        return plf(x, theta, xlim)

    fig, axs = plt.subplots()
    cbar = plot_dkl(plf_adjusted, x, samples, prior_samples, axs, weights = weights)
    cbar = plt.colorbar(cbar, ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(['', r'$1\sigma$', r'$2\sigma$', r'$3\sigma$'])

    if title:
        axs.set_title(title)
    axs.set_ylim(ylim)
    axs.set_ylabel('y')
    axs.set_xlabel('x')

    fig.tight_layout()
    plt.savefig('output.png')

def generate_plot_overlay(file_roots, nDims, xlim, ylim):
    params_list_0, _ = get_params_from_nDims(nDims[0])
    params_list_1, _ = get_params_from_nDims(nDims[1])

    samples_0, weights_0 = samples_from_getdist_chains(params_list_0, file_roots[0])
    samples_1, weights_1 = samples_from_getdist_chains(params_list_1, file_roots[1])

    x = np.linspace(xlim[0], xlim[1], 200)

    fig, axs = plt.subplots()
    cbar = plot_contours(plf, x, samples_0, axs, weights = weights_0, ny = 500)
    cbar = plot_contours(plf, x, samples_1, axs, colors=plt.cm.Blues_r, weights = weights_1, ny = 500)

    cbar = plt.colorbar(cbar, ticks = [0, 1, 2, 3])
    cbar.set_ticklabels(['', r'$1\sigma$', r'$2\sigma$', r'$3\sigma$'])

    axs.set_ylim(ylim)
    axs.set_ylabel('y')
    axs.set_xlabel('x')

    fig.tight_layout()
    plt.savefig('output.png')

def generate_file_root(nInternalPoints, fixed):
    my_str = "output_full"
    
    if fixed:
        my_str = my_str + "_fixed"
    
    my_str = my_str + "/primordial_ps_nInternalPoints{}nLive800_polychord_raw/primordial_ps_nInternalPoints{}nLive800".format(nInternalPoints, nInternalPoints)

    return my_str

def extend_samples(samples):
    N = len(samples)
    nDims_list = np.zeros(N)
    new_samples = samples.copy()
    for i in range(N):
        nDims_list[i] = samples[i].shape[1]
    
    nDims = np.max(nDims_list) # now want to extend all of these to nDims, so that fgivenx is happy with us

    for i in range(N):
        sample = samples[i]
        nDims_current = sample.shape[1]
        nSamples = sample.shape[0]

        extend_by = int((nDims - nDims_current) // 2)

        if extend_by > 0:
            sample_x = sample[:, :nDims_current//2 - 1]
            sample_y = sample[:, nDims_current//2 - 1:]

            sample_x = np.concatenate([sample_x, np.ones(shape = (nSamples, extend_by))], axis = 1)
            sample_y = np.concatenate([sample_y, np.ones(shape = (1, extend_by)) * sample_y[:, -1][:, np.newaxis]], axis = 1)
            new_samples[i] = np.concatenate([sample_x, sample_y], axis = 1)
    return new_samples

