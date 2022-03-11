import numpy as np

def get_input_params_dict(nDims, xlim, ylim, info_params = {}):
    xmin, xmax = xlim
    ymin, ymax = ylim

    assert nDims % 2 == 0
    nPoints = (nDims + 2) // 2

    x_proposal = np.linspace(0, 1, nPoints + 2)
    x_proposal = x_proposal[1:-1]

    for i in range(1, nPoints - 1):
        fraction = i / nPoints
        proposal = xmin * (1-fraction) + xmax * fraction
        info_params['x' + str(i)] = {'prior': {'min': xmin, 'max': xmax}, 'proposal': proposal}
    for i in range(nPoints):
        info_params['y' + str(i)] = {'prior': {'min': ymin, 'max': ymax}}
    return info_params

def update_output(info, nDims):
    try:
        new_output = info['output'] + '_nDims' + \
            str(nDims) + 'nLive' + str(info['sampler']['polychord']['nlive'])
    except KeyError:
        new_output = info['output'] + '_nDims' + str(nDims)

    return new_output
