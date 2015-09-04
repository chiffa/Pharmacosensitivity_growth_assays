__author__ = 'Andrei'

import numpy as np
from chiffatools.Linalg_routines import rm_nans

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
        if val1-val2 > st*percentile_5_range:
            return 1
        if val1-val2 < -st*percentile_5_range:
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


def extract(data_container, cell, drug, drug_versions, cell_index, drug_index):

    def nan(_drug_n):
        return np.all(np.isnan(data_container[cell_n, _drug_n]))

    cell_n = cell_index[cell]
    retained_drugs = [drug_v for drug_v in drug_versions[drug] if not nan(drug_index[drug_v])]

    drugs_nos = [drug_index[drug_v] for drug_v in retained_drugs]

    drug_vals = [data_container[cell_n, drug_n] for drug_n in drugs_nos ]
    drug_c = [drug_v[1]*drug_c_array for drug_v in retained_drugs]

    drug_vals = np.hstack(drug_vals)
    drug_c = np.hstack(drug_c)

    c_argsort = np.argsort(drug_c)

    drug_c = drug_c[c_argsort]
    drug_vals = drug_vals[:, c_argsort, :] # standard error of mean is the standard deviation divided by the sqrt of number of non-nul elements

    return drug_vals, drug_c

def correct_values(raw_values, background, initial):
    # TODO: compute the values that cancel out the noise (subtraction of background)'
    # TODO: compute the growth/death compared to the initial OD
    # TODO: compute the growh/death in terms of signal-to-noise ratio

    # TODO: check if the interplate repeats variance is stronger than intraplate repeate variance

    pass

def compute_stats(values, concentrations):
    unique_values = np.unique(concentrations)

    means = np.zeros_like(unique_values)
    errs = np.zeros_like(unique_values)
    for i, val in enumerate(unique_values):
        mask = concentrations == val
        vals = rm_nans(values[:, mask, :])
        means[i] = np.mean(vals)
        errs[i] = np.std(vals)/np.sqrt(vals.shape[0])

    return means, errs, unique_values
