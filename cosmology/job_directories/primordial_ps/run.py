from cobaya.yaml import yaml_load_file
from cobaya.run import run
from cobaya.theory import Theory
from PowerSpectraModule import return_prior, get_input_params_dict, get_params_from_nDims, power_spectra, update_output
import numpy as np

test = False
debug = False
nDims = 18
xlim = [-4, -0.3]
ylim = [2, 4]
yaml_filename = 'tightpriors.yaml'

class FeaturePrimordialPk(Theory):
    """
        Theory class producing a slow-roll-like power spectrum with an enveloped,
        linearly-oscillatory feture on top.
    """

    params_list, params = get_params_from_nDims(nDims)

    def initialize(self):
        self.ks = np.logspace(xlim[0], xlim[1], 1000)
        self.params_list, params = get_params_from_nDims(nDims)

    def calculate(self, state, want_derived=True, **params_values_dict):

        params_values = [params_values_dict[p] for p in self.params_list]
        ks, Pks = power_spectra(self.ks, params_values, xlim = xlim)
        state['primordial_scalar_pk'] = {
            'k': ks, 'Pk': Pks, 'log_regular': False}

    def get_primordial_scalar_pk(self):
        return self.current_state['primordial_scalar_pk']

my_prior = return_prior(nDims)

info = yaml_load_file(yaml_filename)

info_prior = {"my_prior": my_prior}
info_params = get_input_params_dict(nDims, xlim, ylim, info_params = info['params'])
info_theory = {"my_theory": FeaturePrimordialPk, 'camb': {'external_primordial_pk': True}}
info_output = update_output(info, nDims)

info['prior'] = info_prior
info['params'] = info_params
info['theory'] = info_theory
info['output'] = info_output



updated_info, sampler = run(info, test = test, debug = debug)

# This is some code we've added to see how branch merging works. It serves no puspose beyond that. 