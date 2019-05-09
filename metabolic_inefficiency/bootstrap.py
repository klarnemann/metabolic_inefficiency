import numpy as np

def bootstrap_pct_delta_differences(group1, group2, high_mask, low_mask, tracer, \
                                    voxel_mask=voxel_mask):
    '''
    Computes group difference of the percent difference between two sets of voxels.
    
    Parameters
    ----------
    subjects : list or numpy.ndarray
      list of subject IDs
    high_mask : numpy.ndarray
      boolean array specifying which voxels are expected to have higher values
    low_mask : numpy.ndarray
      boolean array specifying which voxels are expected to have lower values
    tracer : str
      specifies which PET data to load, i.e. pib or tau
    voxel_mask : numpy.ndarray
      boolean array specifying which voxels to extract from 3D PET image    
    '''
    boot_subjects_group1 = np.random.choice(group1, len(group1), replace=True)
    boot_avg_group1 = make_group_average_pet(boot_subjects_group1, tracer, voxel_mask=voxel_mask)
    boot_subjects_group2 = np.random.choice(group2, len(group2), replace=True)
    boot_avg_group2 = make_group_average_pet(boot_subjects_group2, tracer, voxel_mask=voxel_mask)
    boot_avg_diff = boot_avg_group1 - boot_avg_group2
    avg_diff_high = np.mean(np.array(boot_avg_diff)[high_mask])
    avg_diff_low = np.mean(np.array(boot_avg_diff)[low_mask])
    return ((avg_diff_high - avg_diff_low)/((avg_diff_high + avg_diff_low)/2))*100

def run_bootstrap_pct_delta_differences(group1, group2, high_mask, low_mask, tracer, \
                                       voxel_mask=voxel_mask, n_reps=500):
    '''Iteratively runs bootstrap to return expected value and confidence interval of \n
    group difference of the percent difference between two sets of voxels.
    
    Parameters
    ----------
    subjects : list or numpy.ndarray
      list of subject IDs
    high_mask : numpy.ndarray
      boolean array specifying which voxels are expected to have higher values
    low_mask : numpy.ndarray
      boolean array specifying which voxels are expected to have lower values
    tracer : str
      specifies which PET data to load, i.e. pib or tau
    voxel_mask : numpy.ndarray
      boolean array specifying which voxels to extract from 3D PET image
    n_reps : int
      number of iterations for bootstrapping
    '''
    boot_diff = []
    for i in np.arange(n_reps):
        pct_diff = bootstrap_pct_delta_differences(group1, group2, high_mask, low_mask, tracer, voxel_mask)
        boot_diff.append(pct_diff)
    boot_diff.sort()
    print '%s [%s, %s]' % (np.mean(boot_diff), boot_diff[int(n_reps*0.025)], boot_diff[int(n_reps*0.975)])
    return

def bootstrap_pct_delta_differences_based_on_efficiency(group1, group2, residual, tracer, \
                                                  voxel_mask=voxel_mask, n_reps=500):
    '''
    Returns expected value and confidence interval of difference between groups of the percent \n
    difference between metabolically inefficient and efficient brain areas.
    
    Parameters
    ----------
    subjects : list or numpy.ndarray
      list of subject IDs
    residual : numpy.ndarray
      residual error for each voxel
    tracer : str
      specifies which PET data to load, i.e. pib or tau
    voxel_mask : numpy.ndarray
      boolean array specifying which voxels to extract from 3D PET image
    n_reps : int
      number of iterations for bootstrapping
    '''
    ineff_mask = np.array(residual > 0, dtype=bool)
    eff_mask = np.array(residual <= 0, dtype=bool)
    run_bootstrap_pct_delta_differences(group1, group2, ineff_mask, eff_mask, tracer, voxel_mask, n_reps)
    return
    
def bootstrap_pct_delta_differences_based_on_degree(group1, group2, degree, tracer, thr, \
                                              voxel_mask=voxel_mask, n_reps=500):
    '''
    Returns expected value and confidence interval of difference between groups of the percent \n
    difference between hubs and non-hubs.
    
    Parameters
    ----------
    subjects : list or numpy.ndarray
      list of subject IDs
    residual : numpy.ndarray
      residual error for each voxel
    tracer : str
      specifies which PET data to load, i.e. pib or tau
    voxel_mask : numpy.ndarray
      boolean array specifying which voxels to extract from 3D PET image
    n_reps : int
      number of iterations for bootstrapping
    '''
    hubs = np.array(degree >= thr, dtype=bool)
    nonhubs = np.array(degree < thr, dtype=bool)
    run_bootstrap_pct_delta_differences(group1, group2, hubs, nonhubs, tracer, voxel_mask, n_reps)
    return

def bootstrap_pct_differences(subjects, high_mask, low_mask, tracer, voxel_mask=voxel_mask):
    '''
    Computes percent difference between two sets of voxels.
    
    Parameters
    ----------
    subjects : list or numpy.ndarray
      list of subject IDs
    high_mask : numpy.ndarray
      boolean array specifying which voxels are expected to have higher values
    low_mask : numpy.ndarray
      boolean array specifying which voxels are expected to have lower values
    tracer : str
      specifies which PET data to load, i.e. pib or tau
    voxel_mask : numpy.ndarray
      boolean array specifying which voxels to extract from 3D PET image    
    '''
    boot_subjects = np.random.choice(subjects, len(subjects), replace=True)
    boot_avg = make_group_average_pet(boot_subjects, tracer, voxel_mask=voxel_mask)
    avg_high = np.mean(np.array(boot_avg)[high_mask])
    avg_low = np.mean(np.array(boot_avg)[low_mask])
    return ((avg_high - avg_low)/((avg_high + avg_low)/2))*100

def run_boostrap_pct_differences(subjects, high_mask, low_mask, tracer, \
                                 voxel_mask=voxel_mask, n_reps=500):
    '''
    Iteratively runs bootstrap to return expected value and confidence interval of percent difference \
    between two sets of voxels.
    
    Parameters
    ----------
    subjects : list or numpy.ndarray
      list of subject IDs
    high_mask : numpy.ndarray
      boolean array specifying which voxels are expected to have higher values
    low_mask : numpy.ndarray
      boolean array specifying which voxels are expected to have lower values
    tracer : str
      specifies which PET data to load, i.e. pib or tau
    voxel_mask : numpy.ndarray
      boolean array specifying which voxels to extract from 3D PET image
    n_reps : int
      number of iterations for bootstrapping
    '''
    boot_diff = []
    for i in np.arange(n_reps):
        pct_diff = bootstrap_pct_differences(subjects, high_mask, low_mask, tracer, voxel_mask)
        boot_diff.append(pct_diff)
    boot_diff.sort()
    print '%s [%s, %s]' % (np.mean(boot_diff), boot_diff[int(n_reps*0.025)], boot_diff[int(n_reps*0.975)])
    return

def bootstrap_pct_differences_based_on_efficiency(subjects, residual, tracer, \
                                                  voxel_mask=voxel_mask, n_reps=500):
    '''
    Returns expected value and confidence interval of difference between metabolically inefficient \n
    and efficient brain areas.
    
    Parameters
    ----------
    subjects : list or numpy.ndarray
      list of subject IDs
    residual : numpy.ndarray
      residual error for each voxel
    tracer : str
      specifies which PET data to load, i.e. pib or tau
    voxel_mask : numpy.ndarray
      boolean array specifying which voxels to extract from 3D PET image
    n_reps : int
      number of iterations for bootstrapping
    '''
    ineff_mask = np.array(residual > 0, dtype=bool)
    eff_mask = np.array(residual <= 0, dtype=bool)
    run_boostrap_pct_differences(subjects, ineff_mask, eff_mask, tracer, voxel_mask, n_reps)
    return
    
def bootstrap_pct_differences_based_on_degree(subjects, degree, tracer, thr, \
                                              voxel_mask=voxel_mask, n_reps=500):
    '''
    Returns expected value and confidence interval of difference between hubs and non-hubs.
    
    Parameters
    ----------
    subjects : list or numpy.ndarray
      list of subject IDs
    residual : numpy.ndarray
      residual error for each voxel
    tracer : str
      specifies which PET data to load, i.e. pib or tau
    voxel_mask : numpy.ndarray
      boolean array specifying which voxels to extract from 3D PET image
    n_reps : int
      number of iterations for bootstrapping
    '''
    hubs = np.array(degree >= thr, dtype=bool)
    nonhubs = np.array(degree < thr, dtype=bool)        
    run_boostrap_pct_differences(subjects, hubs, nonhubs, tracer, voxel_mask, n_reps)
    return
