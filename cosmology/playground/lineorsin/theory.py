import numpy as np
import scipy

def line(x):
    """ Returns a line(2 pi x) function """

    x_mod = np.mod(x + 0.25, 1)

    y1 = 4 * x_mod - 1
    y2 = 3 - 4 * x_mod

    return np.where(x_mod < 0.5, y1, y2)

def sin2pi(x):
    return np.sin(2 * np.pi * x)

class ModelLikelihood():


    """docstring for ModelLikelihood."""

    def __init__(self, nDims = 6, nDerived = 0, f = line, sigma = 0.05, N = 50, xlim = [0, 1], seed = 0):
        self.nDims = nDims
        self.nDerived = nDerived
        self.f = f
        self.sigma = sigma
        self.N = N
        self.xlim = xlim

        self.generate_data(seed = seed)

    def generate_data(self, seed = None):
        if seed is not None: np.random.seed(seed)

        mean = 0
        X_lower, X_upper = self.xlim

        self.x = np.random.uniform(X_lower, X_upper, self.N)
        self.y = self.f(self.x)

        self.x += np.random.normal(mean, self.sigma, self.N)
        self.y += np.random.normal(mean, self.sigma, self.N)

        # And now crop the data so that everything lies within the bounds of the prior
        indexes = np.argwhere(np.logical_and(np.where(self.x <= X_upper, True, False), np.where(self.x >= X_lower, True, False)))

        self.x = np.take(self.x, indexes)
        self.y = np.take(self.y, indexes)

    def nodes_to_line(self, x, y):
        delta_x = np.diff(x); delta_y = np.diff(y)

        m = delta_y / delta_x

        c = y[:-1] - m * x[:-1]

        return m, c

    def __call__(self, theta):


        # Check that theta values correspond to a valid set of nodes
        nDims = len(theta)
        assert nDims % 2 == 0
        assert nDims == self.nDims

        # Convert from the nodes to values of m and c
        node_x = np.concatenate([np.array([0]), theta[:nDims//2-1], np.array([1])])
        node_y = theta[nDims//2-1:]
        # Check that the x values are sufficiently well spaced, otherwise return a low likelihood
        dx = np.diff(node_x)
        dx_less = np.less_equal(dx, 1e-5)
        if np.any(dx_less):
            return -1e30, []
        # And if that's all fine, then we can return to working out the actual likelihood
        m, c = self.nodes_to_line(node_x, node_y)

        # Construct the constant which comes before the erf integral. Details of this deviration are found on the OneNote page
        # The essense is that since the integral is over a linear function, then we can write it fully in terms of erf, which enables much faster evalutation
        A = -(self.x**2 + self.y**2 + c**2 - 2 * self.y * c - ((self.x + m * self.y - m * c)**2 / (1 + m ** 2))  ) / (2 * self.sigma**2)
        #A = np.exp(A)

        B = 1 / (2 * np.sqrt(2 * np.pi) * self.sigma * (np.max(node_x) - np.min(node_x)) * np.sqrt(1 + m ** 2))

        x_upper = node_x[1:]
        x_lower = node_x[:-1]

        x_upper = np.expand_dims(x_upper, axis = 0) # has shape (1, N-1)
        x_lower = np.expand_dims(x_lower, axis = 0)

        # Find the dimensionless quantities which we evaluate erf over
        translate = (self.x + m * self.y - m * c) / (1 + m**2)
        multiply = np.sqrt(1 + m ** 2) / (np.sqrt(2) * self.sigma)
        u_upper = multiply * (x_upper - translate)
        u_lower = multiply * (x_lower - translate)

        # And calculate the integral
        B = (scipy.special.erf(u_upper) - scipy.special.erf(u_lower)) * B

        # I now has shape (Nj, N-1)
        # We want to sum across N-1 to evaluate the individual integrals for each point, then take logarithm to determine logL, then sum to work out overall logL
        logL = scipy.special.logsumexp(A, b = B, axis = 1)
        logL = np.sum(logL)

        # And now we check for nan, so if nan, then we say this is very unlikely indeed
        # Note we've already checked earlier to make sure that the x coordinates aren't too finely spaced, so this should go some way to resolving this.
        if np.isnan(logL):
            return -1e30, []
        return logL, []

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