import numpy as np
import pandas as pd
import nibabel as ni
from scipy.sparse import csr_matrix

def load_sparse_csr(filename):
    '''
    Returns scipy.sparse_csr data in numpy.matrix format.
    
    Parameters
    ----------
    filename : str
       filename of data in scipy.sparse.csr_matrix format (i.e. *.npz)
    '''
    loader = np.load(filename)
    mat = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
    return mat.todense()

def get_imap_fdg_data(subject, voxel_mask):
    '''
    Returns FDG-PET (i.e. glucose metabolism) data for an IMAP subject.
    
    Parameters
    ----------
    subject : str
        Subject ID
    voxel_mask : array-like, dtype=bool, shape (46, 55, 46)
        Indices of voxels from 3D PET data.
    '''
    imap_fdgdir = '/home/jagust/graph/data/caen/data_YC_Caen/FDG-PET'
    f = '%s/%s/MNI152/subsamp_fdg_suvr_pons_tu_norm.nii.gz'  % (imap_fdgdir, subject)
    if os.path.isfile(f):
        fdg = ni.load(f).get_data()
        return fdg[voxel_mask]
    else:
        print f, 'does not exist'

def get_bacs_pet_data(subject, tracer, voxel_mask):
    '''
    Returns FTP-PET (i.e. tau) data for a BACS subject.
    
    Parameters
    ----------
    subject : str
        Subject ID
    tracer : str
      Which PET data to load. Options: pib, ftp, fdg
    voxel_mask : array-like, dtype=bool, shape (46, 55, 46)
        Indices of voxels from 3D PET data.
    pet_df : pandas.DataFrame
      PET directories for each subject (e.g. )
    '''
    try:
        if tracer == 'fdg':
            closest_fdg_df_f = '/home/jagust/graph/data/fdg/shen_150/closest_fdg.xlsx'
            closest_fdg_df = pd.read_excel(closest_fdg_df_f, index_col=0)
            subdir = closest_fdg_df.closest_fdg_dir.loc[subject]
            dataf = '%s/subsamp_warpd_2MNI152_T1_2mm_brainsuvr_pons.nii.gz' % (subdir)
        elif tracer == 'ftp':
            subdir = '/home/jagust/graph/data/tau/tau_scans/arda/%s' % (subject)
            dataf = '%s/subsamp_suvr_cereg.nii.gz' % (subdir)
        elif tracer == 'pib':
            closest_pib_df_f = '/home/jagust/graph/data/pib/shen_150/closest_pib.xlsx'
            closest_pib_df = pd.read_excel(closest_pib_df_f, index_col=0)
            subdir = closest_pib_df.closest_pib_dir.loc[subject]
            dataf = '%s/subsamp_warpd_2MNI152_T1_2mm_braindvr_cereg.nii.gz' % (subdir)
        data = ni.load(dataf).get_data()
        return data[voxel_mask]
    except ValueError:
        print '%s does not exist' % (globstr)


def make_group_average_pet(subjects, tracer, voxel_mask, groups=None):
    '''
    Loads PET data for each subject and returns average value for each voxel.
    
    Parameters
    ----------
    subjects : list or numpy.ndarray
      List of subject IDs
    tracer : str
      Which PET data to load. Options: pib, ftp, fdg
    voxel_mask : array-like, dtype=bool, shape (46, 55, 46)
        Indices of voxels from 3D PET data.
    group : list or None
      List of the group of each subject (for FDG-PET data)
    
    Returns
    -------
    avg : numpy.ndarray
      average value for each voxel across subjects
    '''
    n_voxels = len(voxel_mask[voxel_mask])
    avg = np.zeros(shape=(len(subjects), n_voxels))
    for i, subject in enumerate(subjects):
        if tracer == 'pib':
            avg[i, :] = get_bacs_pet_data(subject, tracer, voxel_mask)
        elif tracer == 'ftp':
            avg[i, :] = get_bacs_pet_data(subject, tracer, voxel_mask)
        elif tracer == 'fdg':
            group = groups[i]
            if group == 'imap':
                avg[i, :] = get_imap_fdg_data(subject, voxel_mask)
            elif group == 'bacs':
                avg[i, :] = get_bacs_pet_data(subject, tracer, voxel_mask)
    avg = pd.DataFrame(avg)
    return np.array(avg.mean(axis=0, skipna=True), dtype=float).reshape(n_voxels)
