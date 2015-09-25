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
    Deletes an element inside the triplicates if one of them is strongly outlying compared to the others

    :param points:
    :return:
    """
    if all(np.isnan(points)):  # early termination if all points are nan
        return points

    arr_of_interest = pdist(points[:, np.newaxis])
    _min, _max = (np.min(arr_of_interest), np.max(arr_of_interest))
    containment = t.interval(0.95, 1, scale=_min/2)[1]

    if _max > containment:
        outlier = 2 - np.argmin(arr_of_interest)
        msk = np.array([True, True, True])
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


def calculate_information(means, errs):

    def inner_comparison(idx1, idx2):
        return (means[idx1] - means[idx2]) / np.sqrt(errs[idx1]**2 + errs[idx2]**2)

    total_delta = inner_comparison(0, -1)
    high_start = inner_comparison(np.argmax(means), 0)
    low_finish = inner_comparison(-1, np.argmin(means))
    # print 'delta: %s, start: %s, finish: %s, total: %s' % (total_delta, high_start, low_finish,
    #                                                        total_delta - high_start - low_finish)

    return total_delta - high_start - low_finish


def estimate_differences(mean_diff_array):
    mean_diff_array = np.array(mean_diff_array)
    return np.nanstd(mean_diff_array, axis=0, ddof=1)


def correct_plates(plate_stack, concentrations, std_of_tools,
                   replicate_cleaning=True, filter_level=np.nan,
                   info_threshold=3, bang_threshold=20):
    """
    Performs the correction of the plates status

    :param plate:
    :param concentrations:
    :param std_of_tools:
    :return:
    """
    re_plate_stack =  []
    means_stack = []
    errs_stack = []
    unique_concs_stack = []
    
    ghost = np.empty_like(plate_stack[0, :, :])
    ghost.fill(np.nan)

        # removal of outliers in triplicates have to be performed first because they affect stds
    if replicate_cleaning:
        np.apply_along_axis(clean_tri_replicates, 2, plate_stack, std_of_tools)

    for i in range(0, plate_stack.shape[0]):
        plate = plate_stack[i, :, :][np.newaxis, :, :]

        # the plates are not assembled yet.
        # TODO: problem: this collapses the means
        means, errs, stds, freedom_degs, unique_concs = compute_stats(plate, concentrations, std_of_tools)

        flat_ghost = np.empty_like(means)
        flat_ghost.fill(np.nan)

        msk = np.logical_not(np.isnan(means))
        # in this specific case, nan removal is required for info calculation to be properly implemented

        # if not np.isnan(filter_level):
        #   msk = np.logical_and(msk, errs > filter_level)
        # # this was desabled because the masking breaks the indexing routines further down the road

        re_means = means[msk]
        re_stds = stds[msk]
        re_unique_concs = unique_concs[msk]

        for i, conc in enumerate(concentrations): # this clears sets that were filtered out due to excessive noise.
            if conc not in unique_concs:
                plate[:, i, :] = np.nan

        info = calculate_information(re_means, re_stds)
        bang = np.max(re_means)/std_of_tools

        unique_concs_stack.append(unique_concs)
        if info > info_threshold and bang > bang_threshold:
            re_plate_stack.append(plate[0, :, :])
            means_stack.append(means)
            errs_stack.append(errs)

        else:
            re_plate_stack.append(ghost)
            means_stack.append(flat_ghost)
            errs_stack.append(errs)

    # this fragment fails in case we try to filter out additional points from the plot. Hence the off switch above
    re_plate_stack = np.array(re_plate_stack)
    means_stack = np.array(means_stack)
    errs_stack = np.array(errs_stack)
    unique_concs_stack = np.array(unique_concs_stack)

    return re_plate_stack, means_stack, errs_stack, unique_concs_stack


def clean_nans(stake_of_interest, dims=3):
    mask = []
    for i in range(0, stake_of_interest.shape[0]):
        if dims == 3:
            mask.append(np.all(np.isnan(stake_of_interest[i, :, :])))
        if dims == 2:
            mask.append(np.all(np.isnan(stake_of_interest[i, :])))

    mask = np.logical_not(np.array(mask))

    return mask


def normalize(plate_stack, means_stack, errs_stack, std_of_tools, normalization_vector = None):
    if normalization_vector is None:
        normalization_vector = []
        for i in range(0, means_stack[0]):
            normalization_vector.append(means_stack[i, np.logical_not(np.isnan(means_stack[i, :]))][0])  # should be normalization to 0
        normalization_vector = np.array(normalization_vector)

    plate_stack /= normalization_vector[:, np.newaxis, np.newaxis]
    means_stack /= normalization_vector[:, np.newaxis]
    errs_stack /= normalization_vector[:, np.newaxis]
    std_of_tools = std_of_tools / normalization_vector

    return  plate_stack, means_stack, errs_stack, std_of_tools


def combine(plate_stack, concentrations, std_of_tools_vector):
    std_of_tools = np.max(std_of_tools_vector)
    means, errs, stds, freedom_degs, unique_concs = compute_stats(plate_stack, concentrations, std_of_tools)

    return means, errs, unique_concs


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
