import numpy as np
from cobaya.theory import Theory
from helper_functions import plf


class FeaturePrimordialPk(Theory):
    """
        Theory class producing a slow-roll-like power spectrum with an enveloped,
        linearly-oscillatory feture on top.
    """

    params_list, params = get_params_from_nDims(nDims)

    def initialize(self):
        self.ks = np.logspace(-4, -0.3, 1000)

    def calculate(self, state, want_derived=True, **params_values_dict):

        params_values = [params_values_dict[p] for p in params_list]
        ks, Pks = feature_power_spectrum(self.ks, params_values)
        state['primordial_scalar_pk'] = {
            'k': ks, 'Pk': Pks, 'log_regular': False}

    def get_primordial_scalar_pk(self):
        return self.current_state['primordial_scalar_pk']

def plf(x, theta):
    nDims = len(theta)
    node_x = np.concatenate(
        [np.array([0]), theta[:nDims//2 - 1], np.array([1])])
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

def get_input_params_dict(nDims, xlim, ylim):
    xmin, xmax = xlim
    ymin, ymax = ylim

    assert nDims % 2 == 0
    nPoints = (nDims + 2) // 2
    info_params = {}

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

    string = 'lambda ' + x_string + ' : -300 if np.any(np.less_equal(np.diff(np.array([' + x_string + '])), 0)) else 0'

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

def power_spectra(ks, theta):
    log10ks = np.log10(ks)

    lntentenPks = plf(log10ks, theta)

    Pks = 1e-10 * np.exp(lntentenPks)

    return ks, Pks

def get_params_list_dict(self):

    



