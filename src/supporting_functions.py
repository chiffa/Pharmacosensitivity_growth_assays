__author__ = 'Andrei'

import numpy as np
from chiffatools.Linalg_routines import rm_nans
from scipy.stats import t, norm
from scipy.spatial.distance import pdist
import os

drug_c_array = np.array([0]+[2**_i for _i in range(0, 9)])*0.5**8


def safe_dir_create(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def index(myset):
    return dict((elt, i) for i, elt in enumerate(myset))


def broadcast(subline):
    if len(subline) !=30:
        print subline
        raise Exception('wrong number of items in subline')
    else:
        arr = np.array(subline)
        return arr.reshape((10, 3))


def make_comparator(percentile_5_range):
    st = np.sqrt(2)

    def compare(val1, val2):
        if val1 - val2 > st * percentile_5_range:
            return 1
        if val1 - val2 < -st * percentile_5_range:
            return -1
        else:
            return 0

    return compare


def lgi(lst, index_list):
    """
    List get indexes: recovers indexes in the list in the provided index list and returns the result in the form of an
    array

    :param lst:
    :param index_list:
    :return:
    """
    return np.array([lst[i_] for i_ in index_list])


def p_stabilize(array, percentile):
    p_low = np.percentile(rm_nans(array), percentile)
    p_high = np.percentile(rm_nans(array), 100-percentile)
    array[array < p_low] = p_low
    array[array > p_high] = p_high
    return array


def get_boundary_correction(TF, background_std):

    def surviving_fraction(_float):
        return np.ceil(norm.sf(0, _float, background_std)*background_std*1.96)

    surviving_fraction = np.vectorize(surviving_fraction)
    violating_TF_mask = TF < background_std*1.96
    if np.any(violating_TF_mask):
        TF[violating_TF_mask] = surviving_fraction(TF[violating_TF_mask])

    return TF


def get_relative_growth(raw_values, initial, std):
    fold_growth = raw_values - initial[:, :, np.newaxis]
    sigmas = fold_growth / std
    nc_sigmas = raw_values / std

    return fold_growth, sigmas, nc_sigmas


def get_t_distro_outlier_bound_estimation(array, background_std):

    narray = rm_nans(array)

    low, up = t.interval(0.95, narray.shape[0]-1, np.mean(narray), np.sqrt(np.var(narray)+background_std**2))
    up, low = (up-np.mean(narray), np.mean(narray)-low)

    return max(up, low)


def clean_tri_replicates(points, std_of_tools):
    """

    :param points:
    :return:
    """
    if all(np.isnan(points)):
        return points
    arr_of_interest = pdist(points[:, np.newaxis])
    _min, _max = (np.min(arr_of_interest), np.max(arr_of_interest))
    containment = t.interval(0.95, 1, scale=_min/2)[1]

    if _max > containment:
        outlier = 2 - np.argmin(arr_of_interest)
        msk  = np.array([True, True, True])
        msk[outlier] = False
        _mean, _std = (np.mean(points[msk]), np.std(points[msk]))
        containment_2 = t.interval(0.95, 1, loc=_mean, scale=np.sqrt(_std**2+std_of_tools**2))
        if points[outlier] > containment_2[1] or points[outlier] < containment_2[0]:
            points[outlier] = np.nan

    return points


def C0_correction(value_set):
    for i in range(0, value_set.shape[0]):
        if not np.all(np.isnan(value_set)):
            value_set[i, :, :] /= np.nanmean(value_set[i, 0, :])
    return value_set


def compute_stats(values, concentrations, background_std, clean=True):

    unique_values = np.unique(concentrations)

    means = np.zeros_like(unique_values)
    errs = np.zeros_like(unique_values)
    stds = np.zeros_like(unique_values)
    freedom_degs = np.zeros_like(unique_values)
    for i, val in enumerate(unique_values):
        mask = concentrations == val
        vals = rm_nans(values[:, mask, :])
        means[i] = np.mean(vals)
        stds[i] = np.sqrt(np.std(vals)**2 + background_std**2)
        freedom_degs[i] = np.max((vals.shape[0] - 1, 1))
        # errs[i] = stds[i]/np.sqrt(freedom_degs[i])
        errs[i] = get_t_distro_outlier_bound_estimation(vals, background_std)/freedom_degs[i]

    return means, errs, stds, freedom_degs, unique_values


def block_fusion(arg_arr):
    expansion_factor = len(arg_arr)
    ghost = np.empty_like(arg_arr[0])
    ghost.fill(np.nan)
    new_arg_arr = []
    for i, arr in enumerate(arg_arr):
        payload = []
        for j in range(0, i):
            payload.append(ghost)
        payload.append(arr)
        for j in range(i, expansion_factor-1):
            payload.append(ghost)
        new_arg_arr.append(np.vstack(tuple(payload)))

    return np.hstack(new_arg_arr)


def logistic_regression(TF, T0, concentrations, background_std):

    def get_1p_bounds(mean, std, dof):
        return t.interval(0.99, dof, mean, std)

    mask = concentrations == 0.0
    vals_at_0 = rm_nans(TF[:, mask, :])
    max_capacity = get_1p_bounds(np.mean(vals_at_0),
                                 np.sqrt(np.var(vals_at_0) + background_std**2),
                                 vals_at_0.shape[0])[1]*1.05

    compensation_T0 = -np.log2(max_capacity/T0-1)[:, :, np.newaxis]
    compensation_TF = -np.log2(max_capacity/TF-1)

    alphas = compensation_TF - compensation_T0

    return alphas
