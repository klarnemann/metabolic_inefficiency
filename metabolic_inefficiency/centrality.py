import numpy as np
import pandas as pd

def threshold_adjmat_by_cost(mat, cost, rm_nan=False):
    '''Returns a binary connectivity matrix thresholded by the cost.
    
    Parameters
    ----------
    mat : array-like, shape=(nodes,nodes)
        Connectivity matrix.
    cost : float
        Ratio of edges of a graph that will be maintained.
    rm_nan : bool
        Optional: If True remove nan values from the connectivity matrix.
    '''
    n_nodes, _ = mat.shape
    if rm_nan:
        mat[np.isnan(mat)] = 0.
    dat = mat[np.triu_indices(n_nodes, 1)]
    pctl = int((1. - (cost/2.))*100)
    thr = np.percentile(dat, pctl)
    return np.array(mat >= thr, dtype=bool)

def centrality(mat, cost=None, ignore_nan=False):
    '''Returns centrality of a connectivity matrix.
    Note: If cost, computes degree centrality. Otherwise, computes connectivity strength.

    Parameters
    ----------
    mat : array-like, shape=(nodes,nodes)
        Connectivity matrix.
    cost : float
        Optional: Ratio of edges of a graph that will be maintained.
    '''
    if cost:
        bmat = threshold_adjmat_by_cost(mat, cost)
        return bmat.sum(axis=0)
    else:
        if ignore_nan:
            return np.nansum(mat, axis=0)
        else:
            return np.sum(mat, axis=0)

def auc_of_degree_centrality(mat, cost_list, ignore_nan=False):
    '''Returns the AUC of degree centrality across a range of costs.

    Parameters
    ----------
    mat : array-like, shape=(nodes,nodes)
        Connectivity matrix.
    cost_list : list
        List of costs over which AUC will be computed.
    '''
    n_nodes, _ = mat.shape
    deg_df = pd.DataFrame(columns=cost_list)
    for j, cost in enumerate(cost_list):
        deg_df[cost] = centrality(mat, cost)
    if ignore_nan:
        np.nansum(deg_df, axis=1)
    else:
        return deg_df.sum(axis=1)
