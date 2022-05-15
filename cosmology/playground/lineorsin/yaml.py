def get_input_params_dict(nDims, info_params = {}):
    """
        Adds the primordial power spectra parameters to the info dictionary. 

        We pass the x and y values as uniform distributions across the unit hypercube to cobaya
        and subsequently transform in the likelihood to the values of interest. 

        If you have xlims and ylims for the coordinates in the primordial power spectrum, these are 
        dealt with elsewhere.
    """
    assert nDims % 2 == 0
    nPoints = (nDims + 2) // 2

    for i in range(1, nPoints - 1):
        info_params['x' + str(i)] = {'prior': {'min': 0, 'max': 1}}
    for i in range(nPoints):
        info_params['y' + str(i)] = {'prior': {'min': 0, 'max': 1}}
    return info_params