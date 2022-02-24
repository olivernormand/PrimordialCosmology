import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import time


##############################
#### Data generation tools ###
##############################
def line(x):
    """ Returns a line(2 pi x) function """

    x_mod = np.mod(x + 0.25, 1)

    y1 = 4 * x_mod - 1
    y2 = 3 - 4 * x_mod

    return np.where(x_mod < 0.5, y1, y2)

def sin2pi(x):
    return np.sin(2 * np.pi * x)

def generate_noisy_data(f, N, sigma, X_lower = 0, X_upper = 1):
    mean = 0

    x = np.random.uniform(X_lower, X_upper, N)
    y = f(x)

    x += np.random.normal(mean, sigma, N)
    y += np.random.normal(mean, sigma, N)

    # And now crop the data so that everything lies within the bounds

    indexes = np.argwhere(np.logical_and(np.where(x <= X_upper, True, False), np.where(x >= X_lower, True, False)))

    x = np.take(x, indexes)
    y = np.take(y, indexes)

    return x, y

###############################
#### Model Evaluation Tools ###
###############################
def return_piecewise_linear_function(node_x, node_y):

    def piecewise_linear_function(x):

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

    return piecewise_linear_function

def return_linear_function(xi, xi1, yi, yi1):
    def func(x):
        return ( yi * (xi1 - x) + yi1 * (x - xi) ) / (xi1 - xi)
    return func

def full_log_likelihood(x, y, sigma_x, sigma_y, f, X_lower, X_upper):
    if type(sigma_x) is float: sigma_x = np.ones(len(x)) * sigma_x
    if type(sigma_y) is float: sigma_y = np.ones(len(x)) * sigma_x

    # Let's check that the objects are all of the same length
    assert len(x) == len(y) == len(sigma_x) == len(sigma_y)
    j_max = len(x)

    L = 0

    for j in range(j_max):
        L += single_log_likelihood(x[j], y[j], sigma_x[j], sigma_y[j], f, X_lower, X_upper)
    return L

def single_log_likelihood(x, y, sigma_x, sigma_y, f, X_lower, X_upper):
    # This will only compute the integral
    return np.log(quad(integrand, X_lower, X_upper, args = (x, y, sigma_x, sigma_y, f, X_lower, X_upper))[0])

def integrand(z, x, y, sigma_x, sigma_y, f, X_lower, X_upper):

    numerator = np.exp(- (  (x - z)**2 / (2 * sigma_x ** 2)  ) - ( (y - f(z))**2 / (2 * sigma_y ** 2) )  )
    denominator = 2 * np.pi * sigma_x * sigma_y * (X_upper - X_lower)

    return numerator / denominator

if __name__ == '__main__':
    N = 50
    sigma = 0.05
    X_lower = 0
    X_upper = 1

    x1, y1 = generate_noisy_data(sin2pi, N, sigma, X_lower, X_upper)
    x2, y2 = generate_noisy_data(line, N, sigma, X_lower, X_upper)

    print('Shapes', x1.shape, y1.shape, x2.shape, y2.shape)

    fig, axs = plt.subplots()
    axs.scatter(x1, y1)
    axs.scatter(x2, y2)
    fig.savefig('Scatter plot')

    node_x = np.array([0,0.25,0.75, 1])
    node_y = np.array([0, 1, -1, 0])

    trial_function = return_piecewise_linear_function(node_x, node_y)


    t0 = time.time()

    L_sin = full_log_likelihood(x1, y1, sigma, sigma, trial_function, X_lower, X_upper)
    L_linear = full_log_likelihood(x2, y2, sigma, sigma, trial_function, X_lower, X_upper)

    print('Time taken for both evaluations', time.time() - t0)

    posterior_odds_ratio = L_sin - L_linear
    print('Posterior Odds Ratio', posterior_odds_ratio)
