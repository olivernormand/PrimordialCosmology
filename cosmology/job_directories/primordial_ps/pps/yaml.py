import numpy as np

def use_tight_priors(info, n_std = 5, tight = None, fixed = None, exclude = 'mnu'):
    """
        Enables the yaml dictionary to be updated programatically by specifying the fixed and tight priors. 
    """
    info_params = info['params']

    for key in info_params.keys():
        if key in exclude:
            continue
    
        if fixed and key in fixed:
            loc = info_params[key]['ref']['loc']
            info_params[key] = loc
            continue
        
        if tight and key in tight: 
            loc = info_params[key]['ref']['loc']
            std = info_params[key]['ref']['scale']
            info_params[key] = {'prior': {'min': loc - n_std * std, 'max': loc + n_std * std}, 'latex': info_params[key]['latex']}
            continue

        if tight == 'all':
            loc = info_params[key]['ref']['loc']
            std = info_params[key]['ref']['scale']
            info_params[key] = {'prior': {'min': loc - n_std * std, 'max': loc + n_std * std}, 'latex': info_params[key]['latex']}

        if fixed == 'all':
            loc = info_params[key]['ref']['loc']
            info_params[key] = loc

    return info_params

def get_updated_params(nDims, info):
    """
        Adds the primordial power spectra parameters to the info dictionary. 

        We pass the x and y values as uniform distributions across the unit hypercube to cobaya
        and subsequently transform in the likelihood to the values of interest. 

        If you have xlims and ylims for the coordinates in the primordial power spectrum, these are 
        dealt with elsewhere.
    """
    try:
        info_params = info['params']
    except KeyError:
        info_params = {}
    assert nDims % 2 == 0
    nPoints = (nDims + 2) // 2

    for i in range(1, nPoints - 1):
        info_params['x' + str(i)] = {'prior': {'min': 0, 'max': 1}}
    for i in range(nPoints):
        info_params['y' + str(i)] = {'prior': {'min': 0, 'max': 1}}
    return info_params

def get_updated_output(nInternalPoints, info, fixed = None):
    output_str = 'chains/output_full_newer'

    if fixed:
        assert type(fixed) == str # checks against the case where we pass a list
        output_str = output_str + '_fixed_' + fixed + '/primordial_ps'
    else:
        output_str = output_str + '/primordial_ps'
    
    try:
        new_output = output_str + '_nInternalPoints' + \
            str(nInternalPoints) + 'nLive' + str(info['sampler']['polychord']['nlive'])
    except KeyError:
        new_output = output_str + '_nInternalPoints' + str(nInternalPoints)

    print(new_output)
    return new_output
