__author__ = 'Andrei'

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mp
import plot_drawings as PD
from pickle import load
from chiffatools.Linalg_routines import gini_coeff
from chiffatools.dataviz import smooth_histogram
from scipy.stats import norm
import supporting_functions as SF

memdict = load(open('../analysis_runs/memdict.dmp', 'r'))
#[drug, cell_line] -> (means, mean errs, unique_concs), (mean_arr, err_arr, unique, T0)
drug2cell_line = load(open('../analysis_runs/drug2cell_line.dmp', 'r'))
cell_line2drug = load(open('../analysis_runs/cell_line2drug.dmp', 'r'))


def get_concentrations_of_interest(contracted_range=1, base_line=['184A1', '184B5'], err_silencing=True):

    concs_effective_range = []
    all_cell_lines = set()
    stasis_super_pad = []

    for drug, cell_lines in drug2cell_line.items():

        means_pad = []
        errs_pad = []
        unique_c = []
        contractor_pads = []
        stasis_pad = []
        all_cell_lines.update(cell_lines)
        for i, cell_line in enumerate(cell_lines):

            wrap, wrap2 = memdict[drug, cell_line]
            means, errs, unique_c = wrap
            if err_silencing:
                _filter = errs > 0.1
                errs[_filter] = np.nan
                means[_filter] = np.nan
            means_pad.append(means)
            errs_pad.append(errs)

            if contracted_range == 2 :
                if cell_line in base_line:
                    norm_means, norm_errs = wrap2
                    contractor = np.abs(norm_means - 1.)
                    contractor_pads += np.vsplit(contractor, contractor.shape[0])
                    stasis_pad += np.vsplit(norm_means, norm_means.shape[0])

        means_pad = np.array(means_pad)
        errs_pad = np.array(errs_pad)
        if contractor_pads:
            stasis_pad = np.vstack(stasis_pad)
            contractor_pads = np.vstack(contractor_pads)
        else:
            stasis_pad = np.zeros_like(means_pad)
            stasis_pad[:, :] = np.nan
            contractor_pads = np.zeros_like(means_pad)
            contractor_pads[:, :] = np.nan

        _75 = np.nanpercentile(means_pad, 75, axis=0)
        _25 = np.nanpercentile(means_pad, 25, axis=0)

        if contracted_range == 1:
            selector = np.logical_and(_75 < .9, _25 > 0.1 )
            if any(selector):
                sel2 = np.argmin(np.sum(np.isnan(means_pad[:, selector]), axis=0))
                concs_effective_range.append((drug, cell_lines, [unique_c[selector][sel2]],
                                          means_pad[:, selector][:, sel2][:, np.newaxis],
                                          errs_pad[:, selector][:, sel2][:, np.newaxis],
                                          np.nanmean(stasis_pad[:, selector][:, sel2][:, np.newaxis], axis=0)))
        if contracted_range == 2:
            selector = np.logical_and(_25 < .9, _75 > 0.1 )
            if any(selector):
                contractor_pads = contractor_pads[:, selector]
                if not np.all(np.isnan(contractor_pads)):
                    sel3 = np.int(np.median(np.argmin(contractor_pads, axis=1)))
                else:
                    sel3 = 0
                print sel3, np.array(stasis_pad[:, selector][:, sel3])
                stasis_super_pad.append(np.nanmean(stasis_pad[:, sel3]))
                concs_effective_range.append((drug, cell_lines, [unique_c[selector][sel3]],
                                              means_pad[:, selector][:, sel3][:, np.newaxis],
                                              errs_pad[:, selector][:, sel3][:, np.newaxis],
                                              np.nanmean(stasis_pad[:, selector][:, sel3][:, np.newaxis], axis=0)))


        if contracted_range == 3:
            selector = np.logical_and(_75 < .9, _25 > 0.1 )
            if any(selector):
                sel2 = np.argmin(np.sum(np.isnan(means_pad[:, selector]), axis=0))
                concs_effective_range.append((drug, cell_lines, [unique_c[selector][sel2]],
                                          means_pad[:, selector][:, sel2][:, np.newaxis],
                                          errs_pad[:, selector][:, sel2][:, np.newaxis],
                                          np.nanmean(stasis_pad[:, selector][:, sel2][:, np.newaxis], axis=0)))

        if not contracted_range:
            selector = np.logical_and(_25 < .9, _75 > 0.1 )
            if any(selector):
                concs_effective_range.append((drug, cell_lines, unique_c[selector],
                                          means_pad[:, selector],
                                          errs_pad[:, selector],
                                          np.nanmean(stasis_pad[:, selector], axis=0)))

        if contracted_range and not contracted_range in [1, 2, 3]:
            raise Exception('contracted_range value outside [0, 1, 2, 3]')

    return all_cell_lines, concs_effective_range


def stack_data_in_range_of_interest(concs_effective_range):
    all_cell_lines_arr = np.array(list(all_cell_lines))
    means_accumulator = []
    errs_accumulator = []
    names_accumulator = []

    for elt in concs_effective_range:
        print 'anchor!'
        print elt[2]
        print elt[5]
        names_accumulator += [elt[0]+" - %.2E - %.2F"%(conc, stasis) for conc, stasis in zip(elt[2], elt[5])]
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

    return all_cell_lines_arr, means_accumulator, errs_accumulator, names_accumulator

def plot_response(means_accumulator, errs_accumulator, all_cell_lines_arr, names_accumulator, ref_strain='BT483', normalize=False, log=True, sort_by='average'):

    # method 3 & plotting
    all_cell_lines = np.sort(all_cell_lines_arr)

    means_accumulator = means_accumulator.tolist()
    errs_accumulator = errs_accumulator.tolist()
    all_cell_lines = all_cell_lines.tolist()

    idx1 = all_cell_lines.index('184A1')
    idx2 = all_cell_lines.index('184B5')

    mean_for_proxy_WT = np.nanmean(np.array(means_accumulator)[[idx1, idx2], :], axis=0)
    errs_for_proxy_WT = np.nanmean(np.array(errs_accumulator)[[idx1, idx2],:], axis=0)

    all_cell_lines.append('WT_proxy')
    means_accumulator.append(mean_for_proxy_WT.tolist())
    errs_accumulator.append(errs_for_proxy_WT.tolist())

    # support = np.logical_or(
    #                 np.logical_not(np.isnan(means_accumulator[idx1])),
    #                 np.logical_not(np.isnan(means_accumulator[idx2])))

    idx = all_cell_lines.index(ref_strain)
    support = np.logical_not(np.isnan(means_accumulator[idx]))

    average_stress_intesity = np.nanmean(np.array(means_accumulator)[:, support], axis=0)
    log_std_stress_intensity = np.nanstd(np.log2(np.array(means_accumulator)[:, support]), axis=0)
    std_stress_intensity = np.nanstd(np.array(means_accumulator)[:, support], axis=0)
    # argsorter = np.argsort(mean_for_proxy_WT[support]/average_stress_intesity)
    if sort_by == 'average':
        argsorter = np.argsort(average_stress_intesity)
    elif sort_by == 'WT_proxy':
        argsorter = np.argsort(np.array(means_accumulator)[:, support][-1, :])
    elif sort_by == 'ref_strain':
        argsorter = np.argsort(np.array(means_accumulator)[:, support][idx, :])
    else:
        argsorter = np.argsort(average_stress_intesity)

    average_stress_intesity = average_stress_intesity[argsorter]
    log_std_stress_intensity = log_std_stress_intensity[argsorter]
    std_stress_intensity = np.nanstd(np.array(means_accumulator)[:, support], axis=0)

    ramp = np.linspace(0, argsorter.shape[0], argsorter.shape[0]).tolist()
    cmap = mp.cm.get_cmap(name='Paired')

    gini_coeffs = []
    mean_fitness = []

    ln = len(all_cell_lines)
    for i, cell_line in enumerate(all_cell_lines):
        if normalize:
            means_array = np.array(means_accumulator[i])[support][argsorter] / average_stress_intesity
        else:
            means_array = np.array(means_accumulator[i])[support][argsorter]

        errs_array = np.array(errs_accumulator[i])[support][argsorter]

        g_coeff = gini_coeff(means_array)
        support_size = np.sum(np.logical_not(np.isnan(np.array(means_accumulator[i]))))

        if cell_line in ['184A1', '184B5'] or support_size < 10:
            continue

        gini_coeffs.append(g_coeff)
        mean_fitness.append(np.nanmean(means_array))
        if log:
            means_array = np.log2(means_array)
            errs_array = np.log2(1+errs_array)

        if i == idx:
            plt.errorbar(ramp,
                         means_array,
                         yerr=errs_array,
                         label='%s - %.2f - %s'%(cell_line, g_coeff, support_size),
                         color='k')

        else:
            plt.errorbar(ramp,
                         means_array,
                         yerr=errs_array,
                         fmt='.', label='%s - %.2f - %s'%(cell_line, g_coeff, support_size),
                         color=cmap(i/float(ln)))

    mean_g_coeff = gini_coeff(np.array(average_stress_intesity))

    if normalize:
        average_stress_intesity = average_stress_intesity / average_stress_intesity
        std_stress_intensity = std_stress_intensity / average_stress_intesity
    if log:
        average_stress_intesity = np.log(average_stress_intesity)
        std_stress_intensity = log_std_stress_intensity

    plt.plot(ramp,
             average_stress_intesity,
             color = 'r',
             label = '%s - %.2f - %s'%('average', mean_g_coeff , argsorter.shape[0]))
    plt.plot(ramp,
             average_stress_intesity+std_stress_intensity,
             color = 'g')
    plt.plot(ramp,
             average_stress_intesity-std_stress_intensity,
             color = 'g')

    mp.rc('font', size=10)
    plt.xticks(ramp, np.array(names_accumulator)[support][argsorter], rotation='vertical')
    plt.subplots_adjust(bottom=0.25)
    plt.legend(ncol=2)
    plt.show()

    triple_negative = ['BT20', 'BT549', 'HCC1143', 'HCC1187', 'HCC1395', 'HCC1599', 'HCC1806', 'HCC1937', 'HCC2185',
        'HCC3153', 'HCC38', 'HCC70', 'HS578T', 'MDAMB157', 'MDAMB231', 'MDAMB436', 'MDAMB468', 'SUM102PT', 'SUM52PE']


    for i, cell_line in enumerate(all_cell_lines):
        support_size = np.sum(np.logical_not(np.isnan(np.array(means_accumulator[i]))))
        means_array = np.array(means_accumulator[i])[support][argsorter]
        mean_fit = np.nanmean(means_array)
        g_coeff = gini_coeff(means_array)

        if support_size > 15:
            if cell_line in triple_negative:
                plt.plot([g_coeff], [mean_fit],
                     'o', color = 'r',
                     label='%s g: %.2f af: %.2f s: %s' % (cell_line, g_coeff, mean_fit, support_size))
            else:
                plt.plot([g_coeff], [mean_fit],
                     'o', color = 'k',
                     label='%s g: %.2f af: %.2f s: %s' % (cell_line, g_coeff, mean_fit, support_size))

    plt.xlabel('gini coefficient')
    plt.ylabel('average fitness across conditions')
    plt.legend(ncol=2)
    plt.show()

    means_accumulator = np.array(means_accumulator)
    baseline = means_accumulator[idx, :][support][argsorter][np.newaxis, :]
    norm_sorted_means = means_accumulator[:, support][:, argsorter]/baseline
    sorted_names = np.array(names_accumulator)[support][argsorter]

    return norm_sorted_means, sorted_names


def plot_normalized(norm_sorted_means, sorted_names, all_cell_lines):
    accumulator = []
    ramp = np.linspace(0, sorted_names.shape[0], sorted_names.shape[0]).tolist()
    cmap = mp.cm.get_cmap(name='Paired')

    ln = len(all_cell_lines)
    for i, cell_line in enumerate(all_cell_lines):
        support_size = np.sum(np.logical_not(np.isnan(np.array(norm_sorted_means[i]))))

        if support_size < 10:
            continue

        else:
            log_transformed = np.log2(np.array(norm_sorted_means[i]))
            accumulator.append(log_transformed)
            plt.plot(ramp,
                     log_transformed, '.',
                     label='%s'%(cell_line),
                     color=cmap(i/float(ln)))

    accumulator = np.array(accumulator)
    means = np.nanmean(accumulator, axis=0)
    stds = np.nanstd(accumulator, axis=0)

    plt.plot(ramp, means, label='mean')
    plt.plot(ramp, stds, label = 'stds')

    mp.rc('font', size=10)
    plt.xticks(ramp, sorted_names, rotation='vertical')
    plt.subplots_adjust(bottom=0.25)
    # plt.legend(ncol=2)
    plt.show()

    plt.plot(means, stds, 'o')
    plt.xlabel('means')
    plt.ylabel('std')
    plt.show()


def _95p_center(means_accumulator, errs_accumulator):

    def helper(initial, bounds, target):
        shift = target - initial
        if np.abs(shift) < bounds:
            return target
        else:
            scale = bounds / np.abs(shift)
            return initial + shift*scale

    target_percentile = np.sum(np.logical_not(np.isnan(means_accumulator)), axis=0)-1
    target_percentile = 1-np.power(np.ones_like(target_percentile)*0.05, 1/np.sqrt(target_percentile))

    _95p_cosntant = norm.interval(0.95)[1]

    contractor = np.vectorize(lambda x: _95p_cosntant/norm.interval(x)[1])
    contraction_interval = contractor(target_percentile)
    contraction_interval = errs_accumulator/contraction_interval[np.newaxis, :]

    contractor2 = np.vectorize(helper)

    new_means_accumulator = contractor2(means_accumulator, contraction_interval, np.nanmean(means_accumulator, axis=0)[np.newaxis, :])

    return new_means_accumulator


def cell_line_fingerprint(all_cell_lines_arr, means_accumulator, errs_accumulator, names_accumulator):
    all_cell_lines_arr, means_accumulator, errs_accumulator, names_accumulator = SF.preformat(all_cell_lines_arr, means_accumulator, errs_accumulator, names_accumulator)


def drug_fingerprint(all_cell_lines_arr, means_accumulator, errs_accumulator, names_accumulator):
    pass


def drug_combination(all_cell_lines_arr, means_accumulator, errs_accumulator, names_accumulator):
    pass


if __name__ == '__main__':

    all_cell_lines, concs_effective_range = get_concentrations_of_interest(contracted_range=2, err_silencing=True)
    # TODO: add support for range contraction (for isntance for use with T0)

    all_cell_lines_arr, means_accumulator, errs_accumulator, names_accumulator = stack_data_in_range_of_interest(concs_effective_range)

    # means_accumulator = _95p_center(means_accumulator, errs_accumulator)
    norm_sorted_means, sorted_names = plot_response(means_accumulator, errs_accumulator, all_cell_lines_arr, names_accumulator, ref_strain='WT_proxy', normalize=False, log=False, sort_by='WT_proxy')
    plot_normalized(norm_sorted_means, sorted_names, all_cell_lines)