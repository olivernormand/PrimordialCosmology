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

def reverse_fit(t):
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