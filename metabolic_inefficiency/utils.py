import numpy as np
import pandas as pd
import nibabel as ni

def standardize_data(data):
    '''Returns Z-scored data.'''
    return (data - np.mean(data)) / np.std(data)

def normalize_data(data):
    '''Returns normalized data (i.e. sets range of data from 0 to 1).'''
    return ((data - np.min(data)) /  (np.max(data) - np.min(data)))

def make_adjacency_matrix_from_timeseries(ts_f, voxel_mask_f, fill_diagonal=True, upper_triangle=True):
    '''Compute adjacency matrix from 4D timeseries.
    
    ts_f : str
        Filename of 4D nifti timeseries data.
    voxel_mask_f : str
        Filename of 3D nifti voxel mask.
    fill_diagonal : bool
        Optional:  Set diagonal to zero.
    upper_triangle : bool
        Optional: Set lower triangle to zero.
    
    Returns
    -------
    mat : array-like, shape=(voxel,voxel) or (ROI, ROI)'''
    ts = ni.load(ts_f).get_data()# load 4D timeseries
    voxel_mask = ni.load(voxel_mask_f).get_data()# load voxel_mask
    ts_df = pd.DataFrame(ts[mask > 0])# select voxels, format as DataFrame
    mat = np.array(ts_df.T.corr())# pairwise pearson's r
    if fill_diagonal:
        np.fill_diagonal(mat, 0.)# set diagonal to zero
    if upper_triangle:
        mat = np.triu(mat)
    return mat

def threshold_adjmat_by_cost(mat, cost):
    n_nodes, _ = mat.shape
    mat[np.isnan(mat)] = 0.
    dat = mat[np.triu_indices(n_nodes, 1)]
    pctl = int((1. - (cost/2.))*100)
    thr = np.percentile(dat, pctl)
    bmat = np.array(mat >= thr, dtype=bool)
    return bmat

def non_neighbor_mask(voxel, voxel_mask=voxel_mask, n_voxels=n_voxels):
    '''Select all non-neighboring voxels.
    
    Parameters
    ----------
    voxel : int
        index in matrix
    voxel_mask : array-like, dtype=bool, shape (46, 55, 46)
        indices of voxels in an array
    n_voxels : int
        total # of voxels
    
    Returns
    -------
    array-like, dtype-bool, shape (46, 55, 46)'''
    voxel_indices = np.matrix(np.where(voxel_mask > 0)).T
    voxel_indices = voxel_indices[voxel].tolist()[0]
    neighbor_mask = np.zeros(shape = voxel_mask.shape)
    neighbor_mask[voxel_indices[0] - 1 : voxel_indices[0] + 2, \
                  voxel_indices[1] - 1 : voxel_indices[1] + 2, \
                  voxel_indices[2] - 1 : voxel_indices[2] + 2,] = 1.
    neighbor_mask = np.array(neighbor_mask, dtype=bool)
    neighbor_indices = neighbor_mask[voxel_mask]
    return ~neighbor_indices
