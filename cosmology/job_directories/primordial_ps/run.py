import numpy as np

from cobaya.yaml import yaml_load_file
from cobaya.run import run
from cobaya.theory import Theory

from pps.priors import SortedUniformPrior, UniformPrior, hypercube_to_theta
from pps.theory import get_params_from_nDims, power_spectra
from pps.yaml import get_updated_params, get_updated_output, use_tight_priors

def main(test = True, debug = True, resume = False, nInternalPoints = 2, xlim = [-4, -0.3], ylim = [2,4], yaml_filename = 'camb.yaml', tight = None, fixed = None):

    nDims = nInternalPoints * 2 + 2

    x_prior = SortedUniformPrior(xlim[0], xlim[1])
    y_prior = UniformPrior(ylim[0], ylim[1])


    class FeaturePrimordialPk(Theory):
        """
            Theory class defining an arbitrary logarithmic spline.
        """

        params_list, params = get_params_from_nDims(nDims)

        def initialize(self):
            self.ks = np.logspace(xlim[0], xlim[1], 1000)
            self.params_list, params = get_params_from_nDims(nDims)

        def calculate(self, state, want_derived=True, **params_values_dict):

            hypercube = np.array([params_values_dict[p] for p in self.params_list])
            theta = hypercube_to_theta(hypercube, x_prior, y_prior)

            ks, Pks = power_spectra(self.ks, theta, xlim = xlim)
            state['primordial_scalar_pk'] = {
                'k': ks, 'Pk': Pks, 'log_regular': False}

        def get_primordial_scalar_pk(self):
            return self.current_state['primordial_scalar_pk']

    info = yaml_load_file(yaml_filename)

    info_params = use_tight_priors(info, fixed = fixed, tight = tight)
    info_params = get_updated_params(nDims, info)
    info_theory = {"my_theory": FeaturePrimordialPk, 'camb': {'external_primordial_pk': True}}
    info_output = get_updated_output(nInternalPoints, info)

    info['params'] = info_params
    info['theory'] = info_theory
    info['output'] = info_output

    updated_info, sampler = run(info, test = test, debug = debug, resume = resume)

if __name__ == "__main__":
    
    test = True
    debug = False
    resume = False
    nInternalPoints = 6
    xlim = [-4, -0.3]
    ylim = [2, 4]
    yaml_filename = 'camb.yaml'
    tight_priors = 'tau'
    fixed_priors = None
    
    main(test, debug, resume, nInternalPoints, xlim, ylim, yaml_filename, tight_priors, fixed_priors)

