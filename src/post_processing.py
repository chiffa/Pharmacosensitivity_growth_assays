__author__ = 'Andrei'

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mp
import plot_drawings as PD
from pickle import load
from chiffatools.Linalg_routines import gini_coeff

memdict = load(open('../analysis_runs/memdict.dmp', 'r'))
#[drug, cell_line] -> (means, mean errs, unique_concs), (mean_arr, err_arr, unique, T0)
drug2cell_line = load(open('../analysis_runs/drug2cell_line.dmp', 'r'))
cell_line2drug = load(open('../analysis_runs/cell_line2drug.dmp', 'r'))


if __name__ == '__main__':


    # Method 1
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
        if any(selector):
            sel2 = np.argmin(np.sum(np.isnan(means_pad[:, selector]), axis=0))
            concs_effective_range.append((drug, cell_lines, unique_c[selector][sel2],
                                          means_pad[:, selector][:, sel2][:, np.newaxis],
                                          errs_pad[:, selector][:, sel2][:, np.newaxis]))

    # method 2
    all_cell_lines_arr = np.array(list(all_cell_lines))
    means_accumulator = []
    errs_accumulator = []
    names_accumulator = []
    subtitle_accumulator = []

    for elt in concs_effective_range:
        names_accumulator.append(elt[0]+" - %.2E"%elt[2])
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

    # method 3 & plotting
    means_accumulator = np.hstack(tuple(means_accumulator))
    errs_accumulator = np.hstack(tuple(errs_accumulator))
    all_cell_lines = np.sort(all_cell_lines_arr)

    means_accumulator = means_accumulator.tolist()
    errs_accumulator = errs_accumulator.tolist()
    all_cell_lines = all_cell_lines.tolist()
    ramp = np.linspace(0, len(names_accumulator), len(names_accumulator)).tolist()

    idx = all_cell_lines.index('BT483')
    argsorter = np.argsort(np.array(means_accumulator[idx]))
    cmap = mp.cm.get_cmap(name='Paired')
    for i, cell_line in enumerate(all_cell_lines):
        g_coeff = gini_coeff(np.array(means_accumulator[i]))
        support = np.sum(np.logical_not(np.isnan(np.array(means_accumulator[i]))))
        if cell_line in ['184A1', '184B5']:
            plt.errorbar(ramp, np.array(means_accumulator[i])[argsorter],
                         yerr=np.array(errs_accumulator[i])[argsorter],
                         fmt='*', label='%s - %.2f - %s'%(cell_line, g_coeff, support))
        if cell_line == 'BT483':
            plt.errorbar(ramp, np.array(means_accumulator[i])[argsorter],
                         yerr=np.array(errs_accumulator[i])[argsorter],
                         label='%s - %.2f - %s'%(cell_line, g_coeff, support),
                         color='k')
        elif support < 10:
            continue
        else:
            plt.errorbar(ramp, np.array(means_accumulator[i])[argsorter],
                         yerr=np.array(errs_accumulator[i])[argsorter],
                         fmt='o', label='%s - %.2f - %s'%(cell_line, g_coeff, support),
                         color=cmap(i/float(len(names_accumulator))))

    mp.rc('font', size=10)
    plt.xticks(ramp, np.array(names_accumulator)[argsorter], rotation='vertical')
    plt.subplots_adjust(bottom=0.25)
    plt.legend(ncol=2)
    plt.show()

    # TODO: it looks like the gini coefficient is strongly affected by the NaN values of experiments:
    # we need to retain and compare only elements that have a similar support