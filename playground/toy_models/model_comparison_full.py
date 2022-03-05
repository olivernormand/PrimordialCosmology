import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior, SortedUniformPrior
try:
    from mpi4py import MPI
except ImportError:
    pass
from fgivenx import plot_contours, samples_from_getdist_chains
import csv
from scipy.integrate import quad
import time

def line(x):
    """ Returns a line(2 pi x) function """

    x_mod = np.mod(x + 0.25, 1)

    y1 = 4 * x_mod - 1
    y2 = 3 - 4 * x_mod

    return np.where(x_mod < 0.5, y1, y2)

def sin2pi(x):
    return np.sin(2 * np.pi * x)

def prior(hypercube):
    """
        Sorted Uniform Prior on x values
        Uniform Prior on y values
    """
    nDims = len(hypercube)
    assert nDims % 2 == 0
    x_nodes = SortedUniformPrior(0, 1)(hypercube[:nDims//2 - 1])
    y_nodes = UniformPrior(-2, 2)(hypercube[nDims//2 - 1:])

    return np.concatenate([x_nodes, y_nodes])

def dumper(live, dead, logweights, logZ, logZerr):
    print("Last dead point:", dead[-1])

def plf(x, theta):
    nDims = len(theta)
    node_x = np.concatenate([np.array([0]), theta[:nDims//2 - 1], np.array([1])])
    node_y = theta[nDims//2 - 1:]

    N = len(node_x)

    cond_list = []
    func_list = []

    # Generate the condition list
    for i in range(N-1):
        cond_list.append(np.logical_and(np.where(x > node_x[i], True, False), np.where(x < node_x[i+1], True, False)))

    # Generate the function list
    for i in range(N-1):
        func_list.append(return_linear_function(node_x[i], node_x[i+1], node_y[i], node_y[i+1]))

    return np.piecewise(x, cond_list, func_list)

def return_linear_function(xi, xi1, yi, yi1):
    def func(x):
        return ( yi * (xi1 - x) + yi1 * (x - xi) ) / (xi1 - xi)
    return func

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

    def display_data(self):
        fig, axs = plt.subplots()
        axs.scatter(self.x, self.y)
        plt.savefig('data.png')

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

    def old__call__(self, theta):

        # Check that theta values correspond to a valid set of nodes
        nDims = len(theta)
        assert nDims % 2 == 0
        assert nDims == self.nDims

        # Convert from the nodes to values of m and c
        node_x = np.concatenate([np.array([0]), theta[:nDims//2-1], np.array([1])])
        node_y = theta[nDims//2-1:]
        m, c = self.nodes_to_line(node_x, node_y)

        # Construct the constant which comes before the erf integral. Details of this deviration are found on the OneNote page
        # The essense is that since the integral is over a linear function, then we can write it fully in terms of erf, which enables much faster evalutation
        A = -(self.x**2 + self.y**2 + c**2 - 2 * self.y * c - ((self.x + m * self.y - m * c)**2 / (1 + m ** 2))  ) / (2 * self.sigma**2)
        A = np.exp(A)
        A /= 2 * np.sqrt(2 * np.pi) * self.sigma * (np.max(node_x) - np.min(node_x)) * np.sqrt(1 + m ** 2)

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
        I = A * (scipy.special.erf(u_upper) - scipy.special.erf(u_lower))

        # I now has shape (Nj, N-1)
        # We want to sum across N-1 to evaluate the individual integrals for each point, then take logarithm to determine logL, then sum to work out overall logL
        logL = np.sum(I, axis = 1)

        # Account for the small values to avoid runtime and divide by zero errors
        # Smallest possible value is 2.22e-308 but basically we just want it to be small

        logL = np.where(logL <= 1e-300, 1e-300, logL)

        logL = np.log(logL)
        logL = np.sum(logL)

        return logL, []

    def s__call__(self, theta, return_error = False):
        # Testing function which uses the scipy quadrature integration to validate the fast integration tool

        def trial_function(x):
            return plf(x, theta)
        L = 0
        max_frac_error = 0

        N_actual = len(self.x)

        for i in range(N_actual):
            # L += np.log(quad(self.s__call__integrand, self.xlim[0], self.xlim[1], args = (self.x[i], self.y[i], self.sigma, trial_function, self.xlim[0], self.xlim[1]))[0])
            temp = quad(self.s__call__integrand, self.xlim[0], self.xlim[1], args = (self.x[i], self.y[i], self.sigma, trial_function, self.xlim[0], self.xlim[1]))
            trial_frac_error = temp[1] / temp[0]
            if trial_frac_error > max_frac_error:
                max_frac_error = trial_frac_error
            L += np.log(temp[0])

        if return_error:
            return L, max_frac_error
        else:
            return L, []

    def s__call__integrand(self, z, x, y, sigma, f, X_lower, X_upper):
        numerator = np.exp(- (  (x - z)**2 / (2 * sigma ** 2)  ) - ( (y - f(z))**2 / (2 * sigma ** 2) )  )
        denominator = 2 * np.pi * sigma * sigma * (X_upper - X_lower)

        return numerator / denominator

    def lse_comparison(self, theta, printing = True):
        fast = self.old__call__(theta)[0]
        lse = self.__call__(theta)[0]
        slow = self.s__call__(theta)[0]

        frac_diff1 = (lse / fast) - 1
        frac_diff2 = (lse / slow) - 1

        if printing:
            print('Fast {} \t LSE {} \t Slow {} \t FracDiff1 {} \t FracDiff2 {}'.format(fast, lse, slow, frac_diff1, frac_diff2))

    def test_call(self, theta, printing = True):
        # Function that runs tests for the likelihood call
        t0 = time.time()
        fast = self.__call__(theta)[0]
        t_fast = time.time() - t0

        t0 = time.time()
        slow, frac_error = self.s__call__(theta, return_error = True)
        t_slow = time.time() - t0

        frac_diff = (slow / fast) - 1
        if np.abs(frac_diff) < 0.0001:
            valid = True
        else:
            valid = False

        if printing:
            if valid: print('------ VALID ------')
            if not valid: print('------ NOT VALID ------')
            print('Theta {} \t Sigma {}'.format(theta, self.sigma))
            print('Likelihood \t Fast {} \t Slow {} \t Ratio {}'.format(fast, slow, slow / fast))
            print('Time \t \t Fast {} \t Slow {} \t Ratio {}'.format(t_fast, t_slow, round(t_slow / t_fast, 2)))
            print('Fractional Error on Slow Routine \t {}'.format(frac_error))

        return valid

class ModelComparisonRun():

    def __init__(self, nDims, nDerived, nLive, method, N, sigma, seed, plotting = True):
        # Make all objects such that they are 1D numpy arrays
        # Although some we would keep fixed, we will make all arrays for the sake of argument

        self.nDims = self.numpy_compatible(nDims)
        self.nDerived = self.numpy_compatible(nDerived)
        self.nLive = self.numpy_compatible(nLive)
        self.method = self.numpy_compatible(method)
        self.N = self.numpy_compatible(N)
        self.sigma = self.numpy_compatible(sigma)
        self.seed = self.numpy_compatible(seed)
        self.plotting = plotting

        self.n = self.nDims.shape[0] * self.nDerived.shape[0] * self.nLive.shape[0] * self.method.shape[0] * self.N.shape[0] * self.sigma.shape[0] * self.seed.shape[0]
        print("Total number of models: {}".format(self.n))

    def return_evidence(self, file_root, add_chains = True):
        if add_chains:
            file_root = 'chains/' + file_root + '.stats'

        with open(file_root) as file:
            lines = file.readlines()
            values = lines[8].split()
            logZ = round(float(values[2]), 3)
            logZerr = round(float(values[4]), 3)

        return logZ, logZerr

    def numpy_compatible(self, x):
        if type(x) != np.ndarray:
            x = np.array(x)
        if len(x.shape) == 0:
            x = np.expand_dims(x, axis = 0)
        assert len(x.shape) == 1

        return x

    def generate_plot(self, file_root, nDims, f, title = None, add_chains = True, plot_function = True, plot_data = None):
        plotting_file_root = file_root
        if add_chains:
            file_root = 'chains/' + file_root

        samples, weights = samples_from_getdist_chains(['p%i' % i for i in range(nDims)], file_root)

        x = np.linspace(0, 1, 100)

        fig, axs = plt.subplots()
        cbar = plot_contours(plf, x, samples, axs, weights=weights)
        cbar = plt.colorbar(cbar,ticks=[0,1,2,3])
        cbar.set_ticklabels(['',r'$1\sigma$',r'$2\sigma$',r'$3\sigma$'])

        if title: axs.set_title(title)
        axs.set_ylim([-2,2])
        axs.set_ylabel('y')
        axs.set_xlabel('x')

        if plot_function: axs.plot(x, f(x), 'g')
        if plot_data: axs.plot(plot_data[0], plot_data[1], 'b.', markersize = 2)

        fig.tight_layout()
        plt.savefig('output_figures/' + plotting_file_root + '.png')

    def evaluate(self):
        # Determine total number of models to compare
        print('Evaluating evidence for {} models'.format(self.n))

        output_data = [['nDims', 'nDerived', 'nLive', 'method', 'N', 'sigma', 'seed', 'logZ', 'logZerr']]

        # Now how about this for a nested for loop
        for nDims in self.nDims:
            for nDerived in self.nDerived:
                for nLive in self.nLive:
                    for method in self.method:
                        for N in self.N:
                            for sigma in self.sigma:
                                for seed in self.seed:
                                    logZ, logZerr = self.evaluate_individual(nDims, nDerived, nLive, method, N, sigma, seed)
                                    output_data.append([nDims, nDerived, nLive, method, N, sigma, seed, logZ, logZerr])

        # Then save the run
        with open('output.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(output_data)

    def evaluate_individual(self, nDims, nDerived, nLive, method, N, sigma, seed):
        # Ensure that all the objects are integer valued
        nDims = int(nDims)
        nDerived = int(nDerived)
        nLive = int(nLive)
        N = int(N)
        sigma = float(sigma)

        settings = PolyChordSettings(nDims, nDerived)
        settings.nlive = nLive
        settings.do_clustering = True
        settings.read_resume = False

        settings.file_root = '{}nDims{}nLive{}{}N{}sigma'.format(nDims, nLive, method, N, int(sigma * 1000))

        if method == 'Line':
            f = line
        elif method == 'Sin':
            f = sin2pi

        #| Generate the likelihood object
        likelihood = ModelLikelihood(nDims, nDerived, f, sigma, N, seed = seed)

        #| Run PolyChord
        output = pypolychord.run_polychord(likelihood, nDims, nDerived, settings, prior, dumper)

        #| Create a paramnames file and get the evidence
        paramnames = [('p%i' % i, r'\theta_%i' % i) for i in range(nDims)]
        output.make_paramnames_files(paramnames)

        logZ, logZerr = self.return_evidence(settings.file_root)

        #| Now deal with the plotting
        if self.plotting:
            title = '{} Model {} Internal Nodes - logZ = {} +- {}'.format(method, (nDims - 2)//2, logZ, logZerr)
            self.generate_plot(settings.file_root, nDims, f, title = title, add_chains = True, plot_function = True, plot_data = [likelihood.x, likelihood.y])

        return logZ, logZerr

if __name__ == '__main__':
    nDims = [6, 8, 10, 12, 14, 16, 18, 20]
    nDerived = 0
    nLive = 400
    method = ['Line', 'Sin']
    N = 50
    sigma = 0.05
    seed = 2

    run = ModelComparisonRun(nDims, nDerived, nLive, method, N, sigma, seed, plotting = True)
    run.evaluate()
