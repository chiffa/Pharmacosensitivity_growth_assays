__author__ = 'Andrei'

import numpy as np
from chiffatools.Linalg_routines import rm_nans
from scipy.stats import t, norm

drug_c_array = np.array([0]+[2**_i for _i in range(0, 9)])*0.5**8


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
    TF[violating_TF_mask] = surviving_fraction(TF[violating_TF_mask])

    return TF


def correct_values(raw_values, T0_bck, TF_bck, initial, std):
    TF_supressed = raw_values - TF_bck[:, :, np.newaxis]
    TF_supressed = get_boundary_correction(TF_supressed, std)
    T0_supressed = initial - T0_bck
    fold_growth = TF_supressed - T0_supressed[:, :, np.newaxis]
    sigmas = fold_growth / std

    return T0_supressed, TF_supressed, fold_growth, sigmas


def get_t_distro_outlier_bound_estimation(array, background_std):

    narray = rm_nans(array)

    low, up = t.interval(0.95, narray.shape[0]-1, np.mean(narray), np.sqrt(np.var(narray)+background_std**2))
    up, low = (up-np.mean(narray), np.mean(narray)-low)

    # percentiles = 100/narray.shape[0]
    # print narray, narray.shape[0]
    #
    # base = t.interval((100. - 2.*percentiles)/100., narray.shape[0])
    # desired = t.interval(0.95, narray.shape[0]-1)
    #
    # twister = desired[1] / base[1]
    # up = (np.max(narray)-np.mean(narray))*twister
    # low = (np.mean(narray)-np.min(narray))* twister

    return max(up, low)


def compute_stats(values, concentrations, background_std):
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
        errs[i] = get_t_distro_outlier_bound_estimation(vals, background_std)/freedom_degs[i] # TODO: incorporate std error of the data

    return means, errs, stds, freedom_degs, unique_values

def logistics():
    pass

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
