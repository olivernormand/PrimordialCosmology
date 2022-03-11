import numpy as np

from cobaya.yaml import yaml_load_file
from cobaya.run import run
from cobaya.theory import Theory

from lineorsin.priors import SortedUniformPrior, UniformPrior, hypercube_to_theta
from lineorsin.theory import ModelLikelihood, get_params_from_nDims, sin2pi, line
from lineorsin.yaml import get_input_params_dict

test = False
debug = False
resume = True
nDims = 18
xlim = [0, 1]
ylim = [-2, 2]
yaml_filename = 'polychord.yaml'

x_prior = SortedUniformPrior(xlim[0], xlim[1])
y_prior = UniformPrior(ylim[0], ylim[1])
likelihood = ModelLikelihood(nDims = nDims, f = sin2pi, seed = 2, sigma = 0.025)
input_params, params_dict = get_params_from_nDims(nDims)

def my_like(**kwargs):
    hypercube = np.array([kwargs[p] for p in input_params])
    theta = hypercube_to_theta(hypercube, x_prior, y_prior)
    return likelihood(theta)

info = yaml_load_file(yaml_filename)


info_like = {"my_likelihood": {
                "external": my_like,
                "input_params": input_params},
            }
info_params = get_input_params_dict(nDims)


info['likelihood'] = info_like
info['params'] = info_params



updated_info, sampler = run(info, test = test, debug = debug, resume = resume)