from cobaya.run import run
import numpy as np

import numpy as np
from LikelihoodModule import ModelLikelihood, line, sin2pi
from cobaya.yaml import yaml_load_file

# This step is essential to avoid the cobaya not finding the argument self at some point. Makes me very sad.
def my_like(**kwargs):
    theta = [kwargs[p] for p in input_params]
    theta = np.array(theta)
    return likelihood(theta)

def my_prior_old(**kwargs):
    theta = [kwargs[p] for p in input_params]
    nDims = len(theta)

    nX = (nDims + 2) // 2

    x = np.array(theta[:nX])
    x = np.diff(x)
    x = np.less_equal(x, 0)
    x = np.any(x)

    if x:
        return -300
    else:
        return 0

def my_prior_(x0, x1, x2, x3, y1, y2):
    theta = np.array([x0, x1, x2, x3, y1, y2])
    nDims = len(theta)

    nX = (nDims + 2) // 2

    x = np.array(theta[:nX])
    x = np.diff(x)
    x = np.less_equal(x, 0)
    x = np.any(x)

    if x:
        return -300
    else:
        return 0

def my_prior(x1, x2):
    theta = np.array([x1, x2])
    

    x = np.array(theta)
    x = np.diff(x)
    x = np.less_equal(x, 0)
    x = np.any(x)

    if x:
        return -1e30
    else:
        return 0

def generate_lambda(nDims):

    nX = (nDims + 2) // 2
    # Make the string of input parameters, for example x_string = 'x0, x1, x2, x3'
    x_string = ''
    for i in range(1, nX - 1):
        x_string += 'x{}, '.format(i)
    x_string = x_string[:-2]

    string = 'lambda ' + x_string + \
        ' : -1e30 if np.any(np.less_equal(np.diff(np.array([' + x_string + '])), 0)) else 0'

    return eval(string)


nDims = 18
yaml_filename = 'polychord.yaml'

likelihood = ModelLikelihood(nDims=nDims, f = sin2pi, seed=2, sigma=0.025)
input_params = likelihood.get_input_params()
print(input_params)

my_prior = generate_lambda(nDims)

info = yaml_load_file(yaml_filename)

info_like = {"my_likelihood": {
                "external": my_like,
                "input_params": likelihood.get_input_params()},
            } 
            
            
            # "my_prior": {
            #     "external": my_prior, 
            #     "input_params": likelihood.get_input_params()},
            # }

info_prior = {"my_prior": my_prior}
# "input_params": likelihood.get_input_params()}}

info_params = likelihood.get_input_params_dict()

print(info_params)
info['likelihood'] = info_like
info['prior'] = info_prior
info['params'] = info_params


updated_info, sampler = run(info)
