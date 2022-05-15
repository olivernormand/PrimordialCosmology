import matplotlib.pyplot as plt
import numpy as np
from fgivenx import plot_contours, plot_dkl, samples_from_getdist_chains
from fgivenx.drivers import compute_dkl

from pps.theory import get_params_from_nDims, plf
from pps.priors import UniformPrior, SortedUniformPrior, hypercube_to_theta, get_prior_samples, get_prior_weights
from pps.files import run_exists, return_valid_internal_points, generate_file_root, return_evidence

def generate_single_posterior_plot(file_root, nDims, xlim, ylim, fig_title=None, fig_name = None, plot_function = None, axs = None, plot_xlabel = True, plot_ylabel = True, plot_xlabel_top = True, plot_cbar = True, nY = 500):
    """
        Generate posterior plot for given fixed parameters, with options to only label certain axes
    """

    ### Generate the fgivenx plot
    params_list, _ = get_params_from_nDims(nDims)
    samples, weights = samples_from_getdist_chains(params_list, file_root)
    # cache_root = 'cache' + file_root[6:] + "nY{}".format(nY)
    cache_root = 'cache' + file_root[41:] + "nY{}".format(nY)

    x = np.linspace(xlim[0], xlim[1], 200)

    x_prior = SortedUniformPrior(xlim[0], xlim[1])
    y_prior = UniformPrior(ylim[0], ylim[1])

    def plf_adjusted(x, hypercube):
        theta = hypercube_to_theta(hypercube, x_prior, y_prior)
        return plf(x, theta, xlim)

    # Generate fgivenx plot on provided axes, otherwise new axes
    if not axs:
        fig, axs = plt.subplots()
    cbar = plot_contours(plf_adjusted, x, get_prior_samples(samples), axs, colors=plt.cm.Blues_r, lines = False, ny = nY, cache = cache_root + '_prior')
    cbar = plot_contours(plf_adjusted, x, samples, axs, weights=weights, ny = nY, cache = cache_root) # default ny = 500 please
    axs.set_ylim(ylim)
    
    # Optionally add cbar
    if plot_cbar:
        cbar = plt.colorbar(cbar, ticks=[0, 1, 2, 3])
        cbar.set_ticklabels(['', r'$1\sigma$', r'$2\sigma$', r'$3\sigma$'])


    axs = manipulate_axes(axs, dkl = False, plot_xlabel = plot_xlabel, plot_ylabel = plot_ylabel, plot_xlabel_top = plot_xlabel_top)

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

def generate_posterior_plot(nInternalPoints_list, fixed, xlim, ylim, nX = 200, nY = 500, title = None, axs = None, plot_xlabel = True, plot_ylabel = True, plot_xlabel_top = True):
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

    if not axs:
        fig, axs = plt.subplots()
    
    cbar = plot_contours(f, x, prior_samples, axs, logZ = logZ, weights = prior_weights, cache = prior_cache[0], colors=plt.cm.Blues_r, lines = False, ny = nY)
    
    cbar = plot_contours(f, x, samples, axs, logZ = logZ, weights = weights, cache = cache[0], ny = nY)
    axs.set_ylim(ylim)

    axs = manipulate_axes(axs, dkl = False, plot_xlabel = plot_xlabel, plot_ylabel = plot_ylabel, plot_xlabel_top = plot_xlabel_top)

    if title:
        axs.set_title(title)

    plt.savefig(str(fixed) + 'marginalised_posterior.png')

    return cbar

def generate_plots_array(nInternalPoints_list, fixed, xlim, ylim):
    """
        Generate an array of posterior plots, to form final figure for publication
    """
    assert len(nInternalPoints_list) == 8

    nrows = 4
    ncols = 2
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize=(6,8))

    for i in range(nrows):
        for j in range(ncols):
            n = 2 * i + j

            nInternalPoints = nInternalPoints_list[n]
            nDims = nInternalPoints * 2 + 2
            file_root = generate_file_root(nInternalPoints, fixed)
            print(i, j, nInternalPoints)
            plot_xlabel = False 
            plot_ylabel = False 
            plot_xlabel_top = False
            plot_cbar = False 

            
            if i == 3:
                plot_xlabel = True 
            if i == 0:
                plot_xlabel_top = True
            if j == 0:
                plot_ylabel = True

            _, cbar = generate_single_posterior_plot(file_root, nDims, xlim, ylim, fig_name = None, plot_xlabel = plot_xlabel, plot_ylabel = plot_ylabel, plot_xlabel_top = plot_xlabel_top, plot_cbar = False, axs = axs[i, j])

            axs[i,j].set_title("N = {}".format(n), y = 1.0, pad = -14, fontsize = 10)
    
    fig.tight_layout(rect = (0,0,0.96, 1))
    cbar = fig.colorbar(cbar, ax = axs, fraction = 0.03, aspect = 50)
    cbar.ax.tick_params(axis = 'y', which = 'both', direction = 'in')
    cbar.ax.set_title(r'$\sigma$')
    
    plt.savefig('output.png')
    plt.savefig('output.eps', format = 'eps')

def generate_marginalised_posterior_array(nInternalPoints = np.array([0,1,2,3,4,5,6,7]), xlim = [-4, -0.3], ylim = [2,4]):
    """
        Generates an array of marginalised posterior plots
    """

    legends = ['All Free', 'All Fixed', r'Fixed $H_0$', r'Fixed $\Omega_bh^2$', r'Fixed $\Omega_ch^2$', r'Fixed $\tau$']
    parameters = [None, 'all', 'H0', 'ombh2', 'omch2', 'tau']

    nrows = 3
    ncols = 2
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (6, 8))
    for i in range(nrows):
        for j in range(ncols):

            plot_xlabel = False 
            plot_ylabel = False   
            plot_xlabel_top = False
            
            if i == 0:
                plot_xlabel_top = True
            if i == 2:
                plot_xlabel = True
            if j == 0:
                plot_ylabel = True
            
            n = 2 * i + j
            cbar = generate_posterior_plot(nInternalPoints, parameters[n], xlim, ylim, nY = 500, title = legends[n], axs = axs[i,j], plot_xlabel=plot_xlabel, plot_ylabel = plot_ylabel, plot_xlabel_top = plot_xlabel_top)

    cbar = fig.colorbar(cbar, ax = axs, fraction = 0.03, aspect = 50)
    cbar.ax.set_title(r'$\sigma$')

    plt.savefig('marginalised.png')
    plt.savefig('marginalised.eps', format = 'eps')

def generate_dkl(nInternalPoints_list, fixed, xlim, ylim, fig_title = None, fig_name = None, epsilon = 0, rds = True, nX = 100):
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
        file_root = generate_file_root(nInternalPoints, fixed, rds = rds)
        sample, weight = samples_from_getdist_chains(params_list, file_root)
        cache[i] = generate_file_root(nInternalPoints, fixed, cache = True) + '_dkl_{}nX'.format(nX)
        if not rds:
            cache[i] = cache[i] + '_home_'
        prior_cache[i] = cache[i] + '_prior'
        samples[i] = sample 
        weights[i] = weight
        logZ[i] = return_evidence(file_root)[0]

    samples = extend_samples(samples)

    for i in range(N):
        prior_samples[i] = get_prior_samples(samples[i])
        prior_weights[i] = get_prior_weights(weights[i])

    x = np.linspace(xlim[0] + epsilon, xlim[1] - epsilon, nX)

    x_prior = SortedUniformPrior(xlim[0], xlim[1])
    y_prior = UniformPrior(ylim[0], ylim[1])

    def plf_adjusted(x, hypercube):
        theta = hypercube_to_theta(hypercube, x_prior, y_prior)
        return plf(x, theta, xlim)

    f = [plf_adjusted] * N
    dkl = compute_dkl(f, x, samples, prior_samples, logZ = logZ, weights = weights, prior_weights = prior_weights, cache = cache, prior_cache = prior_cache)

    return dkl

def generate_dkl_new(fixed = None, nInternalPoints = np.array([0,1,2,3,4,5,6,7]), xlim =  [-4, -0.3], ylim = [2, 4], nX = 100, rds = True):
    """
        Corrected implementation of the dkl value
    """

    Zs = np.zeros(len(nInternalPoints))
    dkls = np.zeros([len(nInternalPoints), nX])

    for nInternalPoint in nInternalPoints:
        cache = 'cache/dkl_cache{}n_{}'.format(nInternalPoint, str(fixed))
        prior_cache = 'cache/dkl_prior_cache{}n_{}'.format(nInternalPoint, str(fixed))
        x = np.linspace(xlim[0], xlim[1], nX)
        nDims = nInternalPoint * 2 + 2
        x_prior = SortedUniformPrior(xlim[0], xlim[1])
        y_prior = UniformPrior(ylim[0], ylim[1])

        def plf_adjusted(x, hypercube):
            theta = hypercube_to_theta(hypercube, x_prior, y_prior)
            return plf(x, theta, xlim)

        params_list, _ = get_params_from_nDims(nDims)
        file_root = generate_file_root(nInternalPoint, fixed, rds = rds)
        sample, weight = samples_from_getdist_chains(params_list, file_root)
        prior_sample = get_prior_samples(sample); prior_weight = get_prior_weights(weight)



        dkls[nInternalPoint,:] = compute_dkl(f = plf_adjusted, x = x, samples = sample, prior_samples = prior_sample, weights = weight, prior_weights = prior_weight, cache = cache, prior_cache = prior_cache)

        Zs[nInternalPoint] = return_evidence(file_root)[0]

    Zs = Zs - np.max(Zs)
    Zs = np.exp(Zs)
    Zs = Zs / np.sum(Zs)


    dkl = Zs[:, np.newaxis] * dkls

    return np.sum(dkl, axis = 0)

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

    legends = ['All Free', 'All Fixed', r'Fixed $H_0$', r'Fixed $\Omega_bh^2$', r'Fixed $\Omega_ch^2$', r'Fixed $\tau$']
    N = len(fixed)
    dkls = [None] * N
    
    if nInternalPoints is not None:
        nInternalPoints = [nInternalPoints] * N
    else:
        nInternalPoints = return_valid_internal_points(fixed)
        print(nInternalPoints)

    for i in range(N):
        dkls[i] = generate_dkl_new(nInternalPoints = nInternalPoints[i], fixed = fixed[i], xlim = xlim, ylim = ylim)

    fig, axs = plt.subplots()
    x = np.linspace(xlim[0], xlim[1], len(dkls[0]))

    fixed[0] = 'Free'
    for i in range(N):
        axs.plot(x, dkls[i], label = legends[i])

    axs.legend()
    axs.set_xlim(xlim)

    axs = manipulate_axes(axs)
    fig.tight_layout()

    plt.savefig('output.png')
    plt.savefig('output.eps', format = 'eps')

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

def bayes_factor(rds = True):
    """
        Generates a Bayes Factor plot with errors for all models that have a completed cosmology run. 
    """
    nInternalPoints = np.array([0,1,2,3,4,5,6,7])
    parameters = [None, 'all', 'H0', 'ombh2', 'omch2', 'tau']
    legends = ['All Free', 'All Fixed', r'Fixed $H_0$', r'Fixed $\Omega_bh^2$', r'Fixed $\Omega_ch^2$', r'Fixed $\tau$']
    N = len(parameters)
    ylim = [-8, 1]


    logZ = [None] * N
    logZerr = [None] * N

    # Collect the current values of logZ and logZerr
    for i in range(N):
        param = parameters[i]
        logZ_ = []
        logZerr_ = []
        for n in nInternalPoints:
            if run_exists(n, param, rds = rds):
                file_root = generate_file_root(n, param, rds = rds)
                Z, Zerr = return_evidence(file_root)
                logZ_.append(Z)
                logZerr_.append(Zerr)
            else:
                break 
        logZ[i] = np.array(logZ_)
        logZerr[i] = np.array(logZerr_)

    # Now subtract the maximum value for each parameter from eachother. 
    for i in range(N):
        logZ[i] = logZ[i] - np.max(logZ[i])

    # Now plot them all
    fig, axs = plt.subplots()

    for i in range(N):
        n_points = len(logZ[i])
        axs.errorbar(np.arange(n_points), logZ[i], logZerr[i], label = str(legends[i]))

    axs.legend()
    axs.set_ylim(ylim)
    axs.set_ylabel('Bayes Factor')
    axs.set_xlabel('Number of Internal Points N')


    yaxistwin = axs.twinx()
    yaxistwin.set_ylim(np.exp(ylim[0]), np.exp(ylim[1]))
    yaxistwin.set_yscale('log')
    yaxistwin.set_ylabel(r'Relative Evidence $\mathcal{Z}$')

    plt.savefig('output.png')
    plt.savefig('output.eps', format = 'eps')

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

def manipulate_axes(axs, dkl = True, plot_xlabel = True, plot_ylabel = True, plot_xlabel_top = True, xlim = [-4, -0.3]):
    """
        Applies the axes manipulations required to add internal ticks to a plot and correct labeling. 
    """
    # Adjust the xlabels
    xaxistwin = axs.twiny()
    axs.tick_params(axis = 'x', colors = 'white')
    xaxistwin.set_xlim(10**(xlim[0]), 10**(xlim[1]))
    xaxistwin.set_xscale('log')
    xaxistwin.xaxis.set_ticks_position('bottom')
    xaxistwin.tick_params(axis = 'x', which = 'both', bottom = True, direction = 'in', labelbottom = plot_xlabel)

    D_a = 13885
    xaxistwintop = axs.twiny()
    xaxistwintop.set_xlim(10**(xlim[0]) * D_a, 10**(xlim[1]) * D_a)
    xaxistwintop.set_xscale('log')
    xaxistwintop.xaxis.set_ticks_position('top')
    xaxistwintop.tick_params(axis = 'x', which = 'both', top = True, direction = 'in', labeltop = plot_xlabel_top)

    if plot_xlabel: 
        axs.set_xlabel(r"$ k \quad $ [Mpc]$^{-1} $")
    elif not plot_xlabel:
        axs.set_xticks([])

    if plot_xlabel_top:
        xaxistwintop.set_xlabel(r"$\ell$")

    # Adjust the ylabels
    axs.yaxis.set_ticks_position('left')
    axs.tick_params(axis = 'y', which = 'both', left = True, direction = 'in', labelleft = plot_ylabel)
    
    if plot_ylabel:
        if dkl:
            axs.set_ylabel(r"$ D_{KL} \left[ \ln (10^{10} \mathcal{P}_{\mathcal{R}}) \right]$")
        else:
            axs.set_ylabel(r"$ \ln (10^{10} \mathcal{P}_{\mathcal{R}})$")

    return axs