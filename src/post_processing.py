__author__ = 'Andrei'

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mp
import plot_drawings as PD
from pickle import load

memdict = load(open('../analysis_runs/memdict.dmp', 'r'))
#[drug, cell_line] -> (means, mean errs, unique_concs), (mean_arr, err_arr, unique, T0)
drug2cell_line = load(open('../analysis_runs/drug2cell_line.dmp', 'r'))
cell_line2drug = load(open('../analysis_runs/cell_line2drug.dmp', 'r'))

mp.rc('font', size=10)


if __name__ == '__main__':

    concs_effective_range = []
    all_cell_lines = set()

    for drug, cell_lines in drug2cell_line.items():

        cmap = plt.get_cmap('Paired')
        ln = float( len(cell_lines) )
        plt.title(drug)
        means_pad = []
        errs_pad = []
        unique_c = []
        all_cell_lines.update(cell_lines)
        for i, cell_line in enumerate(cell_lines):

            wrap, _ = memdict[drug, cell_line]
            means, errs, unique_c = wrap
            filter = errs > 0.1
            errs[filter] = np.nan
            means[filter] = np.nan
            means_pad.append(means)
            errs_pad.append(errs)

        filter = errs_pad < 0.1
        means_pad = np.array(means_pad)
        errs_pad = np.array(errs_pad)
        _75 = np.nanpercentile(means_pad, 75, axis=0)
        _50 = np.nanpercentile(means_pad, 50, axis=0)
        _25 = np.nanpercentile(means_pad, 25, axis=0)
        selector = np.logical_and(_75 < .8, _25 > 0.1 )
        concs_effective_range.append((drug, cell_lines, unique_c[selector], means_pad[:, selector], errs_pad[:, selector]))

    all_cell_lines_arr = np.array(list(all_cell_lines))
    means_accumulator = []
    errs_accumulator = []
    names_accumulator = []
    subtitle_accumulator = []

    for elt in concs_effective_range:
        names_accumulator += [elt[0]+"%2E"%conc for conc in elt[2]]
        cell_lines = elt[1]
        names_pad = np.array(list(all_cell_lines.difference(set(cell_lines))))
        names_array = np.hstack((np.array(cell_lines), names_pad))
        means = np.pad(elt[3], ((0, names_pad.shape[0]), (0, 0)),
                       mode='constant', constant_values=((np.nan, np.nan), (np.nan, np.nan)))
        errs = np.pad(elt[4], ((0, names_pad.shape[0]), (0, 0)),
                      mode='constant', constant_values=((np.nan, np.nan), (np.nan, np.nan)))

        sorter = np.argsort(names_array)
        means = means[sorter, :]
        errs = errs[sorter, :]

        means_accumulator.append(means)
        errs_accumulator.append(errs)


    means_accumulator = np.hstack(tuple(means_accumulator))
    errs_accumulator = np.hstack(tuple(errs_accumulator))
    all_cell_lines = np.sort(all_cell_lines_arr)

    means_accumulator = means_accumulator.tolist()
    errs_accumulator = errs_accumulator.tolist()
    all_cell_lines = all_cell_lines.tolist()
    ramp = np.linspace(0, len(names_accumulator), len(names_accumulator)).tolist()


    for i, cell_line in enumerate(all_cell_lines):
        plt.errorbar(ramp, means_accumulator[i], yerr=errs_accumulator[i],
                     fmt='o', label=cell_line)

    plt.xticks(ramp, names_accumulator, rotation='vertical')
    plt.legend()
    plt.show()

    # TODO: normalize with respect to the effect before computing the Gini index?
