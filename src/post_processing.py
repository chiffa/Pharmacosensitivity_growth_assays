__author__ = 'Andrei'

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mp
import plot_drawings as PD
from pickle import load
from chiffatools.Linalg_routines import gini_coeff
from chiffatools.dataviz import smooth_histogram

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
        selector = np.logical_and(_25 < .9, _75 > 0.1 )
        if any(selector):
            concs_effective_range.append((drug, cell_lines, unique_c[selector],
                                          means_pad[:, selector],
                                          errs_pad[:, selector]))

            # sel2 = np.argmin(np.sum(np.isnan(means_pad[:, selector]), axis=0))
            # concs_effective_range.append((drug, cell_lines, unique_c[selector][sel2],
            #                               means_pad[:, selector][:, sel2][:, np.newaxis],
            #                               errs_pad[:, selector][:, sel2][:, np.newaxis]))

    # method 2
    all_cell_lines_arr = np.array(list(all_cell_lines))
    means_accumulator = []
    errs_accumulator = []
    names_accumulator = []
    subtitle_accumulator = []

    for elt in concs_effective_range:
        # names_accumulator.append(elt[0]+" - %.2E"%elt[2])
        names_accumulator += [elt[0]+" - %.2E"%conc for conc in elt[2]]
        cell_lines = elt[1]
        names_pad = np.array(list(all_cell_lines.difference(set(cell_lines))))
        names_array = np.hstack((np.array(cell_lines), names_pad))

        print elt[3].shape

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


    idx = all_cell_lines.index('BT483')
    # idx = all_cell_lines.index('HCC1143')
    # idx = all_cell_lines.index('HCC1569')
    # idx = all_cell_lines.index('ZR751')
    # idx = all_cell_lines.index('ZR75B')
    idx1 = all_cell_lines.index('184A1')
    idx2 = all_cell_lines.index('184B5')

    mean_for_proxy_WT = np.nanmean(np.array(means_accumulator)[[idx1, idx2], :], axis=0)
    errs_for_proxy_WT = np.nanmean(np.array(errs_accumulator)[[idx1, idx2],:], axis=0)

    all_cell_lines.append('WT_proxy')
    means_accumulator.append(mean_for_proxy_WT.tolist())
    errs_accumulator.append(errs_for_proxy_WT.tolist())

    # support = np.logical_and(
    #                 np.logical_not(np.isnan(means_accumulator[idx])),
    #                 np.logical_or(
    #                     np.logical_not(np.isnan(means_accumulator[idx1])),
    #                     np.logical_not(np.isnan(means_accumulator[idx2]))))

    # support = np.logical_or(
    #                 np.logical_not(np.isnan(means_accumulator[idx1])),
    #                 np.logical_not(np.isnan(means_accumulator[idx2])))

    support = np.logical_not(np.isnan(means_accumulator[idx]))

    average_stress_intesity = np.nanmean(np.array(means_accumulator)[:, support], axis=0)
    log_std_stress_intensity = np.nanstd(np.log2(np.array(means_accumulator)[:, support]), axis=0)
    # argsorter = np.argsort(mean_for_proxy_WT[support]/average_stress_intesity)
    argsorter = np.argsort(average_stress_intesity)

    average_stress_intesity = average_stress_intesity[argsorter]
    log_std_stress_intensity = log_std_stress_intensity[argsorter]

    ramp = np.linspace(0, argsorter.shape[0], argsorter.shape[0]).tolist()
    cmap = mp.cm.get_cmap(name='Paired')

    gini_coeffs = []
    mean_fitness = []

    for i, cell_line in enumerate(all_cell_lines):
        means_array = np.array(means_accumulator[i])[support][argsorter]
        errs_array = np.array(errs_accumulator[i])[support][argsorter]

        g_coeff = gini_coeff(means_array)
        support_size = np.sum(np.logical_not(np.isnan(np.array(means_accumulator[i]))))

        if cell_line in ['184A1', '184B5'] or support_size < 10:
            continue

        gini_coeffs.append(g_coeff)
        mean_fitness.append(np.nanmean(means_array))
        if i == idx:
            plt.errorbar(ramp,
                         np.log2(means_array),
                         yerr=np.log2(1+errs_array),
                         label='%s - %.2f - %s'%(cell_line, g_coeff, support_size),
                         color='k')

        else:
            plt.errorbar(ramp,
                         np.log2(means_array),
                         yerr=np.log2(1+errs_array),
                         fmt='.', label='%s - %.2f - %s'%(cell_line, g_coeff, support_size),
                         color=cmap(i/float(len(names_accumulator))))

    mean_g_coeff = gini_coeff(np.array(average_stress_intesity))
    plt.plot(ramp,
             np.log(average_stress_intesity),
             color = 'r',
             label = '%s - %.2f - %s'%('average', mean_g_coeff , argsorter.shape[0]))
    plt.plot(ramp,
             np.log(average_stress_intesity)+log_std_stress_intensity,
             color = 'g')
    plt.plot(ramp,
             np.log(average_stress_intesity)-log_std_stress_intensity,
             color = 'g')

    mp.rc('font', size=10)
    plt.xticks(ramp, np.array(names_accumulator)[support][argsorter], rotation='vertical')
    plt.subplots_adjust(bottom=0.25)
    # plt.legend(ncol=2)
    plt.show()

    # smooth_histogram(np.array(gini_coeffs))
    # plt.show()

    # smooth_histogram(np.array(mean_fitness))
    # plt.show()

    # plt.plot(np.array(gini_coeffs), np.array(mean_fitness), 'o')
    # plt.xlabel('gini coefficient')
    # plt.ylabel('average fitness across conditions')
    # plt.show()

    # method 4 and plotting
    base_line = np.array(means_accumulator[idx])[support][argsorter]
    accumulator = []

    for i, cell_line in enumerate(all_cell_lines):

        if cell_line == 'BT483' or support_size < 10:
            continue

        else:
            log_transformed = np.log2(np.array(means_accumulator[i])[support][argsorter]/base_line)
            accumulator.append(log_transformed)
            plt.plot(ramp,
                     log_transformed, '.',
                     label='%s'%(cell_line),
                     color=cmap(i/float(len(names_accumulator))))

    # mp.rc('font', size=10)
    # plt.xticks(ramp, np.array(names_accumulator)[support][argsorter], rotation='vertical')
    # plt.subplots_adjust(bottom=0.25)
    # plt.legend(ncol=2)
    # plt.show()

    accumulator = np.array(accumulator)

    means = np.nanmean(accumulator, axis=0)
    stds = np.nanstd(accumulator, axis=0)

    # sorter = np.argsort(means)
    # means = means[sorter]
    # stds = stds[sorter]

    plt.plot(ramp, means, label='mean')
    plt.plot(ramp, stds, label = 'stds')
    plt.xticks(ramp, np.array(names_accumulator)[support][argsorter], rotation='vertical')
    # plt.legend()
    plt.show()

    plt.plot(means, stds, 'o')
    plt.xlabel('means')
    plt.ylabel('std')
    plt.show()

    # TODO: attention, we need STD of log2-othewise nothing will work properly