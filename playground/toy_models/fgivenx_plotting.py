import numpy as np
import matplotlib.pyplot as plt
from fgivenx import plot_contours, samples_from_getdist_chains

file_root = 'chains/line6nodes_new'
samples, weights = samples_from_getdist_chains(['p%i' % i for i in range(6)], file_root)

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


x = np.linspace(0,1,100)
fig, axs = plt.subplots()
cbar = plot_contours(plf, x, samples, axs, weights=weights)
cbar = plt.colorbar(cbar,ticks=[0,1,2,3])
cbar.set_ticklabels(['',r'$1\sigma$',r'$2\sigma$',r'$3\sigma$'])

axs.set_title('Line2Pi Model 6 Parameter Reconstruction')
axs.set_ylim([-2,2])
axs.set_ylabel('y')
axs.set_xlabel('x')

x_ideal = np.array([0.25, 0.75])
y_ideal = np.array([0, 1, -1, 0])
theta_ideal = np.concatenate([x_ideal, y_ideal])

axs.plot(x, plf(x, theta_ideal), 'g')


fig.tight_layout()
plt.savefig('test3.png')
