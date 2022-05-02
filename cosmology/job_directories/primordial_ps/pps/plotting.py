import matplotlib.pyplot as plt
import numpy as np
from fgivenx import plot_contours, plot_dkl, samples_from_getdist_chains
from fgivenx.drivers import compute_dkl

from pps.theory import get_params_from_nDims, plf
from pps.priors import UniformPrior, SortedUniformPrior, hypercube_to_theta, get_prior_samples, get_prior_weights


def generate_single_posterior_plot(file_root, nDims, xlim, ylim, fig_title=None, fig_name = None, plot_function = None, axs = None, plot_xlabel = True, plot_ylabel = True, plot_cbar = True):
    """
        Generate posterior plot for given fixed parameters, with options to only label certain axes
    """

    ### Generate the fgivenx plot
    params_list, _ = get_params_from_nDims(nDims)
    samples, weights = samples_from_getdist_chains(params_list, file_root)

    x = np.linspace(xlim[0], xlim[1], 200)

    x_prior = SortedUniformPrior(xlim[0], xlim[1])
    y_prior = UniformPrior(ylim[0], ylim[1])

    def plf_adjusted(x, hypercube):
        theta = hypercube_to_theta(hypercube, x_prior, y_prior)
        return plf(x, theta, xlim)

    # Generate fgivenx plot on provided axes, otherwise new axes
    if not axs:
        fig, axs = plt.subplots()
    cbar = plot_contours(plf_adjusted, x, get_prior_samples(samples), axs, colors=plt.cm.Blues_r, lines = False, ny = 500)
    cbar = plot_contours(plf_adjusted, x, samples, axs, weights=weights, ny = 500) # default ny = 500 please
    axs.set_ylim(ylim)
    
    # Optionally add cbar
    if plot_cbar:
        print('here')
        cbar = plt.colorbar(cbar, ticks=[0, 1, 2, 3])
        cbar.set_ticklabels(['', r'$1\sigma$', r'$2\sigma$', r'$3\sigma$'])


    # Optionally add xlabels including logticks
    xaxistwin = axs.twiny()
    axs.tick_params(axis = 'x', colors = 'white')
    xaxistwin.set_xlim(10**(-4), 10**(-0.3))
    xaxistwin.set_xscale('log')
    xaxistwin.xaxis.set_ticks_position('bottom')
    xaxistwin.tick_params(axis = 'x', which = 'both', bottom = True, direction = 'in', labelbottom = plot_xlabel)
    
    if plot_xlabel: 
        axs.set_xlabel(r"$ k \quad $ [Mpc]$^{-1} $")
    elif not plot_xlabel:
        axs.set_xticks([])

    # Optionally add ylabels
    axs.yaxis.set_ticks_position('left')
    axs.tick_params(axis = 'y', which = 'both', left = True, direction = 'in', labelleft = plot_ylabel)
    if plot_ylabel:
        axs.set_ylabel(r"$ \ln (10^{10} \mathcal{P}_{\mathcal{R}})$")

    # Optionally add title
    if fig_title:
        axs.set_title(fig_title)
    
    # Optionally plot the true function
    if plot_function:
        axs.plot(x, plot_function(x), 'g')

    try:
        fig.tight_layout()
    except UnboundLocalError:
        pass


    if fig_name:
        plt.savefig(fig_name)
    return axs, cbar

def generate_plots(nInternalPoints_list, fixed_list, xlim, ylim, single_fig = False):
    """
        Generate multiple posterior plots simultaneously
    """

    for nInternalPoints in nInternalPoints_list:
        
        nDims = nInternalPoints * 2 + 2
        
        for fixed in fixed_list:
            file_root = generate_file_root(nInternalPoints, fixed)
            title = "nInternalPoints = {}".format(nInternalPoints)
            name = "output_figures/nInt{}fixed{}".format(nInternalPoints, int(fixed))
            generate_single_posterior_plot(file_root, nDims, xlim, ylim, fig_title = title, fig_name = name)

def generate_posterior_plot(nInternalPoints_list, fixed, xlim, ylim, nX = 200, nY = 500):
    """
        Generates a posterior plot marginalised over all parameters. 
    """
    N = len(nInternalPoints_list)

    nDims_list = nInternalPoints_list * 2 + 2
    samples = [None] * N
    weights = [None] * N
    prior_samples = [None] * N
    prior_weights = [None] * N
    logZ = [None] * N
    cache = [None] * N 
    prior_cache = [None] * N

    for i in range(N):
        nInternalPoints = nInternalPoints_list[i]
        nDims = nDims_list[i]

        params_list, _ = get_params_from_nDims(nDims)
        file_root = generate_file_root(nInternalPoints, fixed)
        sample, weight = samples_from_getdist_chains(params_list, file_root)
        cache[i] = generate_file_root(nInternalPoints, fixed, cache = True)
        prior_cache[i] = cache[i] + '_prior'
        samples[i] = sample 
        weights[i] = weight
        logZ[i] = return_evidence(file_root)[0]

    samples = extend_samples(samples)

    for i in range(N):
        prior_samples[i] = get_prior_samples(samples[i])
        prior_weights[i] = get_prior_weights(weights[i])

    x = np.linspace(xlim[0], xlim[1], nX)

    x_prior = SortedUniformPrior(xlim[0], xlim[1])
    y_prior = UniformPrior(ylim[0], ylim[1])

    def plf_adjusted(x, hypercube):
        theta = hypercube_to_theta(hypercube, x_prior, y_prior)
        return plf(x, theta, xlim)

    f = [plf_adjusted] * N
    fig, axs = plt.subplots()
    
    cbar = plot_contours(f, x, prior_samples, axs, logZ = logZ, weights = prior_weights, cache = prior_cache[0], colors=plt.cm.Blues_r, lines = False, ny = nY)
    
    cbar = plot_contours(f, x, samples, axs, logZ = logZ, weights = weights, cache = cache[0], ny = nY)
    axs.set_ylim(ylim)

    axs = manipulate_axes(axs, dkl = False)

    plt.savefig(str(fixed) + 'marginalised_posterior.png')



def generate_plots_array(nInternalPoints_list, fixed, xlim, ylim):
    """
        Generate an array of posterior plots, to form final figure for publication
    """
    assert len(nInternalPoints_list) == 8

    nrows = 4
    ncols = 2
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize=(8.27,11.69))

    for i in range(nrows):
        for j in range(ncols):
            n = 2 * i + j

            nInternalPoints = nInternalPoints_list[n]
            nDims = nInternalPoints * 2 + 2
            file_root = generate_file_root(nInternalPoints, fixed)
            print(i, j, nInternalPoints)
            plot_xlabel = False 
            plot_ylabel = False 
            plot_cbar = False 

            if i == 3:
                plot_xlabel = True 
            if j == 0:
                plot_ylabel = True

            _, cbar = generate_single_posterior_plot(file_root, nDims, xlim, ylim, fig_name = None, plot_xlabel = plot_xlabel, plot_ylabel = plot_ylabel, plot_cbar = False, axs = axs[i, j])
    
    cbar = fig.colorbar(cbar, ax = axs, fraction = 0.03, aspect = 50)
    cbar.ax.tick_params(axis = 'y', which = 'both', direction = 'in')
    #fig.tight_layout()
    plt.savefig('output.png')

def generate_dkl(nInternalPoints_list, fixed, xlim, ylim, fig_title = None, fig_name = None, epsilon = 0.001):
    """
        Generate the dkl values marginalised over the nInternalPoints_list for a given set of fixed parameters
    """
    N = len(nInternalPoints_list)

    nDims_list = nInternalPoints_list * 2 + 2
    samples = [None] * N
    weights = [None] * N
    prior_samples = [None] * N
    prior_weights = [None] * N
    logZ = [None] * N
    cache = [None] * N 
    prior_cache = [None] * N

    for i in range(N):
        nInternalPoints = nInternalPoints_list[i]
        nDims = nDims_list[i]
        params_list, _ = get_params_from_nDims(nDims)
        file_root = generate_file_root(nInternalPoints, fixed)
        sample, weight = samples_from_getdist_chains(params_list, file_root)
        cache[i] = generate_file_root(nInternalPoints, fixed, cache = True)
        prior_cache[i] = cache[i] + '_prior'
        samples[i] = sample 
        weights[i] = weight
        logZ[i] = return_evidence(file_root)[0]

    samples = extend_samples(samples)

    for i in range(N):
        prior_samples[i] = get_prior_samples(samples[i])
        prior_weights[i] = get_prior_weights(weights[i])

    x = np.linspace(xlim[0] + epsilon, xlim[1] - epsilon, 100)

    x_prior = SortedUniformPrior(xlim[0], xlim[1])
    y_prior = UniformPrior(ylim[0], ylim[1])

    def plf_adjusted(x, hypercube):
        theta = hypercube_to_theta(hypercube, x_prior, y_prior)
        return plf(x, theta, xlim)

    f = [plf_adjusted] * N
    dkl = compute_dkl(f, x, samples, prior_samples, logZ = logZ, weights = weights, prior_weights = prior_weights, cache = cache, prior_cache = prior_cache)

    return dkl

def generate_single_dkl_plot(file_root, nDims, xlim, ylim, fig_title = None, fig_name = None, epsilon = 0.001):
    """
        Generates a single dkl plot, without marginalisation. Used to test and debug the more complicated tool
    """
    params_list, _ = get_params_from_nDims(nDims)
    sample, weight = samples_from_getdist_chains(params_list, file_root)
    prior_sample = get_prior_samples(sample)
    print(sample.shape, prior_sample.shape, weight.shape)

    x = np.linspace(xlim[0] + epsilon, xlim[1], 100)
    x_prior = SortedUniformPrior(xlim[0], xlim[1])
    y_prior = UniformPrior(ylim[0], ylim[1])

    def plf_adjusted(x, hypercube):
        theta = hypercube_to_theta(hypercube, x_prior, y_prior)
        return plf(x, theta, xlim)

    fig, axs = plt.subplots()
    plot_dkl(plf_adjusted, x, sample, prior_sample, axs, weights = weight)

    if fig_title:
        axs.set_title(fig_title)
    plt.savefig('new_output.png')

def generate_dkl_plot(fixed = [None, 'tau'], nInternalPoints = None, xlim = [-4, -0.3], ylim = [2,4]):
    """
        Generates a dkl plot looping over InternalPoints and fixed parameters. Plots used in final publication. 
    """

    N = len(fixed)
    dkls = [None] * N
    
    if nInternalPoints:
        nInternalPoints = [nInternalPoints] * N
    else:
        nInternalPoints_None = np.array([0,1,2,3,4,5,6,7])
        nInternalPoints_tau = np.array([0,1,2,3,4,5,6,7])
        nInternalPoints = [nInternalPoints_None, nInternalPoints_tau]

    for i in range(N):
        dkls[i] = generate_dkl(nInternalPoints[i], fixed[i], xlim, ylim)

    fig, axs = plt.subplots()
    x = np.linspace(xlim[0], xlim[1], len(dkls[0]))

    fixed[0] = 'Free'
    for i in range(N):
        axs.plot(x, dkls[i], label = fixed[i])

    axs.legend()

    axs = manipulate_axes(axs)

    plt.savefig('output.png')

def generate_plot_overlay(file_roots, nDims, xlim, ylim):
    """
        Archive function no longer used
    """
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

def generate_file_root(nInternalPoints, fixed = None, cache = False):
    """
        Generates the file root according to the (somewhat dated) naming convention. So that files can be referenced simply with nInternalPoints, which parameter is fixed, and whether it is the cache or not. 
    """
    my_str = "chains/output_full"

    if cache:
        my_str = "cache/output_full"
    
    if fixed:
        my_str = my_str + "_fixed_" + fixed
    
    my_str = my_str + "/primordial_ps_nInternalPoints{}nLive800_polychord_raw/primordial_ps_nInternalPoints{}nLive800".format(nInternalPoints, nInternalPoints)

    return my_str

def extend_samples(samples):
    """
        Given a list of samples with different dimensionalities, it forces the samples to have the same dimensionality so that fgivenx will play nice with them. 
    """
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

def return_evidence(file_root):
    """
        Given a file root, will look up the model evidence and return the evidence and its error
    """
    file_root = file_root + '.stats'
    with open(file_root) as file:
        lines = file.readlines()
        values = lines[8].split()
        logZ = round(float(values[2]), 3)
        logZerr = round(float(values[4]), 3)

    return logZ, logZerr

def manipulate_axes(axs, dkl = True):
    """
        Applies the axes manipulations required to add internal ticks to a plot and correct labeling. 
        
        TODO make this compatible with optional x and y labelling
    """
    # Adjust the xlabels
    xaxistwin = axs.twiny()
    axs.tick_params(axis = 'x', colors = 'white')
    xaxistwin.set_xlim(10**(-4), 10**(-0.3))
    xaxistwin.set_xscale('log')
    xaxistwin.xaxis.set_ticks_position('bottom')
    xaxistwin.tick_params(axis = 'x', which = 'both', bottom = True, direction = 'in', labelbottom = True)
    axs.set_xlabel(r"$ k \quad $ [Mpc]$^{-1} $")

    # Adjust the ylabels
    axs.yaxis.set_ticks_position('left')
    axs.tick_params(axis = 'y', which = 'both', left = True, direction = 'in', labelleft = True)
    
    if dkl:
        axs.set_ylabel(r"$ D_{KL} \left[ \ln (10^{10} \mathcal{P}_{\mathcal{R}}) \right]$")
    else:
        axs.set_ylabel(r"$ \ln (10^{10} \mathcal{P}_{\mathcal{R}})$")

    return axs