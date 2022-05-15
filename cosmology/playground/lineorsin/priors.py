import numpy as np

"""
    We define functions to help with implementing sorted uniform priors. 
    Central to this is the forced identifiability transform, which enables us to go from the
    uniform hypercube to our sorted parameters. We define this in the forwards and backwards direction. 
"""

def forward_fit(x):
    """
        Takes unsorted values x from the uniform hypercube between 0 and 1
        Returns sorted values t between 0 and 1
    """
    N = len(x)
    t = np.zeros(N)
    t[N-1] = x[N-1]**(1./N)
    for n in range(N-2, -1, -1):
        t[n] = x[n]**(1./(n+1)) * t[n+1]
    return t

def inverse_fit(t):
    """
        Takes sorted values t between 0 and 1
        Returns unsorted values x from the uniform hypercube between 0 and 1
    """
    N = len(t)
    x = np.zeros(N)
    x[N-1] = t[N-1]**N
    for n in range(N-2, -1, -1):
        x[n] = (t[n] / t[n+1]) ** (n+1)
    return x

class UniformPrior:
    """
        Define a set of transforms between the unit hypercube x and dimensional parameters theta
    """
    def __init__(self, a, b):
        self.a = a 
        self.b = b

    def __call__(self, x):
        # x --> theta
        return self.a + (self.b - self.a) * x

    def inverse(self, theta):
        # theta --> x
        return (theta - self.a) / (self.b - self.a)


class SortedUniformPrior(UniformPrior):
    """
        Extend the Uniform Prior to a Sorted Uniform prior through the forced identifiability transform
        We introduce the set of parameters t which are sorted, and define transforms between x and theta while imposing the ordering constraint. 
    """
    def __call__(self, x):
        t = forward_fit(x)
        return super(SortedUniformPrior, self).__call__(t)
    
    def inverse(self, theta):
        t = super(SortedUniformPrior, self).inverse(theta)
        return inverse_fit(t)

def hypercube_to_theta(hypercube, x_transform, y_transform):
    nDims = len(hypercube)
    assert nDims % 2 == 0
    x_nodes = x_transform(hypercube[:nDims//2 - 1])
    y_nodes = y_transform(hypercube[nDims//2 - 1:])

    return np.concatenate([x_nodes, y_nodes])

def theta_to_hypercube(theta, x_inverse_transform, y_inverse_transform):
    
    nDims = len(theta)
    assert nDims % 2 == 0
    x_nodes = x_inverse_transform(theta[:nDims//2 - 1])
    y_nodes = y_inverse_transform(theta[nDims//2 - 1:])

    return np.concatenate([x_nodes, y_nodes])


