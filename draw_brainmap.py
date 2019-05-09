import numpy as np
import nibabel as ni

def draw_brainmap(input1, voxel_mask, mnif, savef, fill_val=0.0):
    '''
    Create a brainmap, where values in mask are set to input1.
    
    Parameters
    ----------
    input1 : array-like, shape (n_voxels)
        Data to be projected onto 3D brainmap.
    mnif : str
        Filename of MNI template.
    voxel_mask : array-like, dtype=bool, shape (46, 55, 46)
        Indices of voxels in 3D brainmap.
    savef : str
        Filename of nifti data.
    fill_val : float
        Optional: Value set to voxels outside the mask (default is 0).
    '''
    mni = ni.load(mnif)
    mni_aff = mni.get_affine()
    mni_shape = mni.shape
    mni = mni.get_data()
    brainmap = np.ones(shape=(mni_shape))*fill_val
    brainmap[voxel_mask] = input1
    brainmap_nii = ni.Nifti1Image(brainmap, mni_aff)
    brainmap_nii.to_filename(savef)
