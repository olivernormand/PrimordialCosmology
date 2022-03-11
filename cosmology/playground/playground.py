from cobaya.run import run
import numpy as np

import numpy as np
from lineorsin import ModelLikelihood, line, sin2pi
from cobaya.yaml import yaml_load_file

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

info_params = likelihood.get_input_params_dict()

print(info_params)
info['likelihood'] = info_like
info['prior'] = info_prior
info['params'] = info_params


updated_info, sampler = run(info)
