import numpy as np
from scipy.optimize import curve_fit

def power_func(x, a, p):
    '''
    Power function of x, given parameters a and p.
    
    Parematers
    ----------
    x : numpy.ndarray, shape (n_voxels)
      data
    a : float
      parameter
    p : float
      parameter'''
    return a * x**p

def power_model(x_vars, y_vars, power_func):
    '''Returns optimal parameters for power model of data.
    
    x_vars : array-like, shape (n_voxels)
        Training data
    y_vars : array-like, shape (n_voxels)
        Target values
    power_func : function
        Power function of x_vars, given a and p.'''
    x_vars = np.array(x_vars, dtype=np.float64)
    y_vars = np.array(y_vars, dtype=np.float64)
    power_popt, _ = curve_fit(power_func, x_vars, y_vars)#fit power model
    return power_popt

def leave_one_out_power_model(x_vars, y_vars, power_func=power_func, exclude_neighbors=True):
    '''Iteratively runs power model, using a leave-one-subject-out approach.
    
    x_vars : array-like, shape (n_voxels)
        Training data
    y_vars : array-like, shape (n_voxels)
        Target values
    power_func : function
        Power function of x_vars, given a and p.
    exclude_neighbors : bool
        If true, exclude neighboring voxels from training data.'''
    assert (len(x_vars) == len(y_vars)), 'Variables must have same length'
    n_voxels = len(x_vars)
    inds = np.arange(n_voxels)
    # initiate output
    loo_prediction = np.zeros(n_voxels)
    loo_resid = np.zeros(n_voxels)
    loo_a = np.zeros(n_voxels)
    loo_p = np.zeros(n_voxels)
    for i in np.arange(n_voxels):
        if exclude_neighbors:
            inds_mask = non_neighbor_mask(i)# remove to-be predicted voxel and its neighbors
        else:
            inds_mask = inds != i# remove to-be predicted voxel
        p_opt = power_model(x_vars, y_vars, power_func)# fit model
        # get model parameters
        loo_a[i] = p_opt[0]
        loo_p[i] = p_opt[1]
        loo_prediction[i] = power_func(x_vars[i], *p_opt)# predicted values
        loo_resid[i] = y_vars[i] - loo_prediction[i]# residual error
    return loo_prediction, loo_resid, loo_a, loo_p
