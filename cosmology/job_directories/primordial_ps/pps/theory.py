import numpy as np

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
            np.where(x >= node_x[i], True, False), np.where(x <= node_x[i+1], True, False)))

    # Generate the function list
    for i in range(N-1):
        func_list.append(return_linear_function(
            node_x[i], node_x[i+1], node_y[i], node_y[i+1]))

    return np.piecewise(x, cond_list, func_list)

def return_linear_function(xi, xi1, yi, yi1):
    def func(x):
        # Check for overlapping nodes and deal with those as necessary
        if xi1 == xi: # then the nodes are on top of one another
            return np.ones(len(x)) * 0.5 * (yi + yi1)
        
        return (yi * (xi1 - x) + yi1 * (x - xi)) / (xi1 - xi)
    return func

def get_params_from_nDims(nDims):
    """
        Returns the parameter dictionary necessary to get parameters from getdist chains
    """
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

    return ks, Pks

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

