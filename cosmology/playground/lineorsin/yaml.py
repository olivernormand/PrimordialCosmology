def get_input_params(self):
    nDims = self.nDims
    assert nDims % 2 == 0
    nPoints = (nDims + 2) // 2
    input_params = []
    for i in range(1, nPoints - 1):
        input_params.append('x' + str(i))
    for i in range(nPoints):
        input_params.append('y' + str(i))
    return input_params

def get_input_params_dict(self):
    nDims = self.nDims
    xmin, xmax = self.xlim
    ymin, ymax = self.ylim

    assert nDims % 2 == 0
    nPoints = (nDims + 2) // 2
    info_params = {}

    for i in range(1, nPoints - 1):
        info_params['x' + str(i)] = {'prior': {'min': xmin, 'max': xmax}}
    for i in range(nPoints):
        info_params['y' + str(i)] = {'prior': {'min': ymin, 'max': ymax}}
    return info_params