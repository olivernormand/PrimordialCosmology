import numpy as np
from cobaya.theory import Theory

def get_params_from_nDims(nDims):
    params_list = []
    params_dict = {}
    nX = (nDims + 2) // 2

    for i in range(1, nX - 1):
        add_me = 'x{}'.format(i)
        params_list.append(add_me)
        params_dict[add_me] = None
    for i in range(nX):
        add_me = 'y{}'.format(i)
        params_list.append(add_me)
        params_dict[add_me] = None

    return params_list, params_dict


def power_spectrum(As, ns, A, l, c, w,
                           kmin=1e-6, kmax=10,  # generous, for transfer integrals
                           k_pivot=0.05, n_samples_wavelength=20):
    
    # Ensure thin enough sampling at low-k
    delta_k = min(0.0005, l / n_samples_wavelength)
    ks = np.arange(kmin, kmax, delta_k)
    def power_law(k): return As * (k / k_pivot) ** (ns - 1)
    def DeltaP_over_P(k): return (
        A * feature_envelope(k, c, w) * np.sin(2 * np.pi * k / l))
    Pks = power_law(ks) * (1 + DeltaP_over_P(ks))
    return ks, Pks


class FeaturePrimordialPk(Theory):
    """
    Theory class producing a slow-roll-like power spectrum with an enveloped,
    linearly-oscillatory feture on top.
    """

    params_list, params = get_params_from_nDims(nDims)

    n_samples_wavelength = 20
    k_pivot = 0.05

    def calculate(self, state, want_derived=True, **params_values_dict):
        params_values = [params_values_dict[p] for p in params_list]
        ks, Pks = feature_power_spectrum(params_values)
        state['primordial_scalar_pk'] = {'k': ks, 'Pk': Pks, 'log_regular': False}

    def get_primordial_scalar_pk(self):
        return self.current_state['primordial_scalar_pk']


def logprior_high_k(A, c, w, k_high=0.25, A_min=5e-3):
    """
    Returns -inf whenever the feature acts at too high k's only, i.e. such that the
    product of amplituce and evenlope at `k_high` is smaller than `A_min`, given that the
    envelope is centred at `k > k_high`.
    """
    if c < k_high:
        return 0
    return 0 if A * feature_envelope(k_high, c, w) > A_min else -np.inf
