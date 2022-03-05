import numpy as np


def plf(x, theta, xlim = [-4, -0.3]):
    x_lower, x_upper = xlim
    nDims = len(theta)
    node_x = np.concatenate(
        [np.array([x_lower]), theta[:nDims//2 - 1], np.array([x_upper])])
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
        info_params['x' + str(i)] = {'prior': {'min': xmin, 'max': xmax}}
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

    return eval(string)

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
