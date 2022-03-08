import numpy as np
from fgivenx import plot_contours, samples_from_getdist_chains
import matplotlib.pyplot as plt

def plf(x, theta, xlim = [-4, -0.3]):
    """
        Return the piecewise linear function defined by a set of parameters theta

        theta represents the x coordinates of all but the edge points, and the y coordinates of those same points. 
        The x coordinates of the edge points do not need to be included. 

        x represents the values over which you wish the function to be evaluated. 

    """

    x_lower, x_upper = xlim
    nDims = len(theta)
    node_x = np.concatenate([np.array([x_lower]), theta[:nDims//2 - 1], np.array([x_upper])])
    node_y = theta[nDims//2 - 1:]

    N = len(node_x)

    cond_list = []
    func_list = []

    # Generate the condition list
    for i in range(N-1):
        cond_list.append(np.logical_and(
            np.where(x > node_x[i], True, False), np.where(x < node_x[i+1], True, False)))

    # Generate the function list
    for i in range(N-1):
        func_list.append(return_linear_function(
            node_x[i], node_x[i+1], node_y[i], node_y[i+1]))

    return np.piecewise(x, cond_list, func_list)

def return_linear_function(xi, xi1, yi, yi1):
    def func(x):
        return (yi * (xi1 - x) + yi1 * (x - xi)) / (xi1 - xi)
    return func

def get_input_params(nDims):
    assert nDims % 2 == 0
    nPoints = (nDims + 2) // 2
    input_params = []
    for i in range(1, nPoints - 1):
        input_params.append('x' + str(i))
    for i in range(nPoints):
        input_params.append('y' + str(i))
    return input_params

def get_input_params_dict(nDims, xlim, ylim, info_params = {}):
    xmin, xmax = xlim
    ymin, ymax = ylim

    assert nDims % 2 == 0
    nPoints = (nDims + 2) // 2

    x_proposal = np.linspace(0, 1, nPoints + 2)
    x_proposal = x_proposal[1:-1]

    for i in range(1, nPoints - 1):
        fraction = i / nPoints
        proposal = xmin * (1-fraction) + xmax * fraction
        info_params['x' + str(i)] = {'prior': {'min': xmin, 'max': xmax}, 'proposal': proposal}
    for i in range(nPoints):
        info_params['y' + str(i)] = {'prior': {'min': ymin, 'max': ymax}}
    return info_params

def return_prior(nDims):
    nX = (nDims + 2) // 2
    # Make the string of input parameters, for example x_string = 'x0, x1, x2, x3'
    x_string = ''
    for i in range(1, nX - 1):
        x_string += 'x{}, '.format(i)
    x_string = x_string[:-2]

    string = 'lambda ' + x_string + ' : -1e30 if np.any(np.less_equal(np.diff(np.array([' + x_string + '])), 0)) else 0'

    return string

def get_params_from_nDims(nDims):
    params_list = []
    params_dict = {}
    nX = (nDims + 2) // 2

    for i in range(1, nX - 1):
        add_me = 'x{}'.format(i)
        params_list.append(add_me)
        params_dict[add_me] = None
    for i in range(nX):
        add_me = 'y{}'.format(i)
        params_list.append(add_me)
        params_dict[add_me] = None

    return params_list, params_dict

def power_spectra(ks, theta, xlim):
    log10ks = np.log10(ks)

    lntentenPks = plf(log10ks, theta, xlim = xlim)

    Pks = 1e-10 * np.exp(lntentenPks)

    # print('printing', np.mean(log10ks), np.mean(lntentenPks), np.mean(ks), np.mean(Pks))

    return ks, Pks

def generate_plot(file_root, nDims, xlim, ylim, title=None, plot_function=None):
    params_list, _ = get_params_from_nDims(nDims)

    samples, weights = samples_from_getdist_chains(params_list, file_root)

    x = np.linspace(xlim[0], xlim[1], 200)

    fig, axs = plt.subplots()
    cbar = plot_contours(plf, x, samples, axs, weights=weights, ny = 500)
    cbar = plt.colorbar(cbar, ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(['', r'$1\sigma$', r'$2\sigma$', r'$3\sigma$'])

    if title:
        axs.set_title(title)
    axs.set_ylim(ylim)
    axs.set_ylabel('y')
    axs.set_xlabel('x')

    if plot_function:
        axs.plot(x, plot_function(x), 'g')

    fig.tight_layout()
    plt.savefig('output.png')

def true_power_spectra(x):
    """
        Returns the true power spectra using theoretical values of the As and ns to calculate the power spectra in this plane
    """
    y0 = 3.2552
    y1 = 2.970
    x0 = -4
    x1 = -0.3

    dx = x1 - x0 
    dy = y1 - y0 
    m = dy / dx 

    return m * x + y0 - m * x0

def update_output(info, nDims):
    try:
        new_output = info['output'] + '_nDims' + \
            str(nDims) + 'nLive' + str(info['sampler']['polychord']['nlive'])
    except KeyError:
        new_output = info['output'] + '_nDims' + str(nDims)

    return new_output
