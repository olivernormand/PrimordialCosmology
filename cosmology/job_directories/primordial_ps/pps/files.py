import numpy as np
from os.path import exists 

def generate_file_root(nInternalPoints, fixed = None, cache = False, rds = True, exists = False):
    """
        Generates the file root according to the (somewhat dated) naming convention. So that files can be referenced simply with nInternalPoints, which parameter is fixed, and whether it is the cache or not. 
    """
    my_str = "chains/output_full_new"

    if rds:
        my_str = "/rds/user/ocn22/hpc-work/cosmology/chains/output_full"

    if cache:
        my_str = "cache/output_full"
    
    if fixed:
        my_str = my_str + "_fixed_" + fixed
    
    my_str = my_str + "/primordial_ps_nInternalPoints{}nLive800".format(nInternalPoints)

    if exists:
        return my_str
    else:
        return my_str + "_polychord_raw/primordial_ps_nInternalPoints{}nLive800".format(nInternalPoints)


def return_complete_runs(nInternalPoints = np.array([0,1,2,3,4,5,6,7]), parameters = [None, 'H0', 'ombh2', 'omch2', 'tau', 'all'], rds = True, evidence = False):

    for param in parameters:
        for n in nInternalPoints:
            root = generate_file_root(n, param, rds = rds, exists = True)
            completed_root = root + '.logZ'
            started_root = root + '.input.yaml'
            completed_root_exists = exists(completed_root)
            started_root_exists = exists(started_root)
            if completed_root_exists:
                if evidence:
                    logZ, logZerr = return_evidence(generate_file_root(n, param, rds = rds, exists = False))
                    print(param, n, logZ, logZerr)
                else:
                    print(param, n)
            elif started_root_exists:
                print('Started', param, n)
        print()

def run_exists(nInternalPoints, parameter, rds = True):
    root = generate_file_root(nInternalPoints, parameter, rds = rds, exists = True) + '.logZ'
    return exists(root)

def return_valid_internal_points(parameters):
    N = len(parameters)
    nInternalPoints = [None] * N 

    for i in range(N):
        for j in range(8):
            if run_exists(j, parameters[i]):
                nInternalPoints[i] = np.arange(j + 1)
    return nInternalPoints

def return_evidence(file_root):
    """
        Given a file root, will look up the model evidence and return the evidence and its error
    """
    file_root = file_root + '.stats'
    with open(file_root) as file:
        lines = file.readlines()
        values = lines[8].split()
        logZ = round(float(values[2]), 3)
        logZerr = round(float(values[4]), 3)

    return logZ, logZerr