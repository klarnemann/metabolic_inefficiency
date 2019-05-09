import numpy as np

def standardize_data(data):
    '''Returns Z-scored data.'''
    return (data - np.mean(data)) / np.std(data)

def normalize_data(data):
    '''Returns normalized data (i.e. sets range of data from 0 to 1).'''
    return ((data - np.min(data)) /  (np.max(data) - np.min(data)))

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
