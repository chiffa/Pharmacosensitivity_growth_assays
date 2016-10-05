import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mp
import plot_drawings as PD
from pickle import load
from chiffatools.linalg_routines import gini_coeff
from chiffatools.dataviz import smooth_histogram
from scipy.stats import norm
import supporting_functions
from chiffatools.linalg_routines import hierchical_clustering

memdict = load(open('../analysis_runs/memdict.dmp', 'r'))
# [drug, cell_line] -> (means, mean errs, unique_concs), (mean_arr, err_arr, unique, T0)
drug2cell_line = load(open('../analysis_runs/drug2cell_line.dmp', 'r'))
cell_line2drug = load(open('../analysis_runs/cell_line2drug.dmp', 'r'))


def get_concentrations_of_interest(contracted_range=1,
                                   base_line=['184A1', '184B5'],
                                   err_silencing=True):

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

            if contracted_range == 2:
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
            inhib_int_sel = np.logical_and(_75 < .9, _25 > 0.1)
            if any(inhib_int_sel):
                cut_off_2 = np.argmin(np.sum(np.isnan(means_pad[:, inhib_int_sel]), axis=0))

                concs_effective_range.append((drug,
                                              cell_lines,
                                              [unique_c[inhib_int_sel][cut_off_2]],
                                              means_pad[:, inhib_int_sel][:, cut_off_2][:, np.newaxis],
                                              errs_pad[:, inhib_int_sel][:, cut_off_2][:, np.newaxis],
                                              np.nanmean(stasis_pad[:, inhib_int_sel][:, cut_off_2][:, np.newaxis], axis=0)))

        if contracted_range == 2:
            inhib_int_sel = np.logical_and(_75 < .9, _25 > 0.1)
            # inhib_int_sel = np.nanmedian(stasis_pad, axis=0) > 0.9
            if any(inhib_int_sel):
                contractor_pads = contractor_pads[:, inhib_int_sel]
                if not np.all(np.isnan(contractor_pads)):
                    cut_off_3 = np.int(np.median(np.argmin(contractor_pads, axis=1)))
                else:
                    cut_off_3 = 0
                stasis_super_pad.append(np.nanmean(stasis_pad[:, cut_off_3]))
                concs_effective_range.append((drug,
                                              cell_lines,
                                              [unique_c[inhib_int_sel][cut_off_3]],
                                              means_pad[:, inhib_int_sel][:, cut_off_3][:, np.newaxis],
                                              errs_pad[:, inhib_int_sel][:, cut_off_3][:, np.newaxis],
                                              np.nanmean(stasis_pad[:, inhib_int_sel][:, cut_off_3][:, np.newaxis], axis=0)))

        if contracted_range == 3:
            inhib_int_sel = np.logical_and(_75 < .9, _25 > 0.1)
            if any(inhib_int_sel):
                cut_off_2 = np.argmin(np.sum(np.isnan(means_pad[:, inhib_int_sel]), axis=0))
                concs_effective_range.append((drug,
                                              cell_lines,
                                              [unique_c[inhib_int_sel][cut_off_2]],
                                              means_pad[:, inhib_int_sel][:, cut_off_2][:, np.newaxis],
                                              errs_pad[:, inhib_int_sel][:, cut_off_2][:, np.newaxis],
                                              np.nanmean(stasis_pad[:, inhib_int_sel][:, cut_off_2][:, np.newaxis], axis=0)))

        if not contracted_range:
            inhib_int_sel = np.logical_and(_25 < .9, _75 > 0.1)
            if any(inhib_int_sel):
                concs_effective_range.append((drug,
                                              cell_lines,
                                              unique_c[inhib_int_sel],
                                              means_pad[:, inhib_int_sel],
                                              errs_pad[:, inhib_int_sel],
                                              np.nanmean(stasis_pad[:, inhib_int_sel], axis=0)))

        if contracted_range and contracted_range not in [1, 2, 3]:
            raise Exception('contracted_range value outside [0, 1, 2, 3]')

    return all_cell_lines, concs_effective_range


def stack_data_in_range_of_interest(concs_effective_range):
    _all_cell_lines_arr = np.array(list(all_cell_lines))
    _means_accumulator = []
    _errs_accumulator = []
    _names_accumulator = []

    for elt in concs_effective_range:
        _names_accumulator += [elt[0]+" - %.2E - %.2F" % (conc, stasis) for conc, stasis in zip(elt[2], elt[5])]
        cell_lines = elt[1]
        names_pad = np.array(list(all_cell_lines.difference(set(cell_lines))))
        # TODO: this piece of sorting seems to be working, even if it is not necessarily
        # working pretty well or clearly. Refactor if have time
        names_array = np.hstack((np.array(cell_lines), names_pad))

        means = np.pad(elt[3], ((0, names_pad.shape[0]), (0, 0)),
                       mode='constant', constant_values=((np.nan, np.nan), (np.nan, np.nan)))
        errs = np.pad(elt[4], ((0, names_pad.shape[0]), (0, 0)),
                      mode='constant', constant_values=((np.nan, np.nan), (np.nan, np.nan)))

        sorter = np.argsort(names_array)
        means = means[sorter, :]
        errs = errs[sorter, :]

        _means_accumulator.append(means)
        _errs_accumulator.append(errs)

    _means_accumulator = np.hstack(tuple(_means_accumulator))
    _errs_accumulator = np.hstack(tuple(_errs_accumulator))

    return _all_cell_lines_arr, _means_accumulator, _errs_accumulator, _names_accumulator


def plot_response(_means_accumulator, _errs_accumulator, _all_cell_lines_arr, _names_accumulator,
                  ref_strain='BT483', normalize=False, log=True, sort_by='average'):

    _means_accumulator = _means_accumulator.tolist()
    _errs_accumulator = _errs_accumulator.tolist()
    _all_cell_lines = _all_cell_lines_arr.tolist()

    idx1 = _all_cell_lines.index('184A1')
    idx2 = _all_cell_lines.index('184B5')

    mean_for_proxy_wt = np.nanmean(np.array(_means_accumulator)[[idx1, idx2], :], axis=0)
    errs_for_proxy_wt = np.nanmean(np.array(_errs_accumulator)[[idx1, idx2], :], axis=0)

    _all_cell_lines.append('WT_proxy')
    _means_accumulator.append(mean_for_proxy_wt.tolist())
    _errs_accumulator.append(errs_for_proxy_wt.tolist())

    # support = np.logical_or(
    #                 np.logical_not(np.isnan(_means_accumulator[idx1])),
    #                 np.logical_not(np.isnan(_means_accumulator[idx2])))

    ref_strain_idx = _all_cell_lines.index(ref_strain)
    support = np.logical_not(np.isnan(_means_accumulator[ref_strain_idx]))

    average_stress_intesity = np.nanmean(np.array(_means_accumulator)[:, support], axis=0)
    log_std_stress_intensity = np.nanstd(np.log2(np.array(_means_accumulator)[:, support]), axis=0)
    std_stress_intensity = np.nanstd(np.array(_means_accumulator)[:, support], axis=0)
    # argsorter = np.argsort(mean_for_proxy_wt[support]/average_stress_intesity)

    if sort_by == 'average':
        argsorter = np.argsort(average_stress_intesity)
    elif sort_by == 'WT_proxy':
        argsorter = np.argsort(np.array(_means_accumulator)[:, support][-1, :])
    elif sort_by == 'ref_strain':
        argsorter = np.argsort(np.array(_means_accumulator)[:, support][ref_strain_idx, :])
    else:
        argsorter = np.argsort(average_stress_intesity)

    average_stress_intensity = average_stress_intesity[argsorter]
    log_std_stress_intensity = log_std_stress_intensity[argsorter]
    std_stress_intensity = np.nanstd(np.array(_means_accumulator)[:, support], axis=0)

    ramp = np.linspace(0, argsorter.shape[0], argsorter.shape[0]).tolist()
    cmap = mp.cm.get_cmap(name='Paired')

    gini_coeffs = []
    mean_fitness = []

    # this is a padded plot. As such it needs to be factored out.

    ln = len(_all_cell_lines)
    for i, cell_line in enumerate(_all_cell_lines):
        if normalize:
            means_array = (np.array(_means_accumulator[i])[support] / average_stress_intesity)[argsorter]
            errs_array = (np.array(_errs_accumulator[i])[support] / average_stress_intesity)[argsorter]
        else:
            means_array = np.array(_means_accumulator[i])[support][argsorter]
            errs_array = np.array(_errs_accumulator[i])[support][argsorter]

        g_coeff = gini_coeff(means_array)
        support_size = np.sum(np.logical_not(np.isnan(np.array(_means_accumulator[i]))))

        if cell_line in ['184A1', '184B5'] or support_size < 10:
            continue

        gini_coeffs.append(g_coeff)
        mean_fitness.append(np.nanmean(means_array))

        if log:
            means_array = np.log2(means_array)
            errs_array = np.log2(1+errs_array)

        if i == ref_strain_idx:
            plt.errorbar(ramp,
                         means_array,
                         yerr=errs_array,
                         label='%s - %.2f - %s' % (cell_line, g_coeff, support_size),
                         color='k')

        else:
            plt.errorbar(ramp,
                         means_array,
                         yerr=errs_array,
                         fmt='.', label='%s - %.2f - %s' % (cell_line, g_coeff, support_size),
                         color=cmap(i/float(ln)))

    mean_g_coeff = gini_coeff(np.array(average_stress_intesity))

    if normalize:
        average_stress_intesity = average_stress_intesity / average_stress_intesity
        std_stress_intensity = std_stress_intensity / average_stress_intesity

    if log:
        average_stress_intesity = np.log(average_stress_intesity)
        std_stress_intensity = log_std_stress_intensity

    plt.plot(ramp,
             average_stress_intesity[argsorter],
             color='r',
             label='%s - %.2f - %s' % ('average', mean_g_coeff, argsorter.shape[0])
             )

    plt.plot(ramp,
             (average_stress_intesity + std_stress_intensity)[argsorter],
             color='g')

    plt.plot(ramp,
             (average_stress_intesity - std_stress_intensity)[argsorter],
             color='g')

    mp.rc('font', size=10)
    plt.xticks(ramp, np.array(_names_accumulator)[support][argsorter], rotation='vertical')
    plt.subplots_adjust(bottom=0.25)
    plt.legend(ncol=2)
    plt.show()

    # This is a 2-high-light, gini-mean plotting procedure. It needs to be factored out

    high_lighted_1 = ['WT_proxy', '184A1', '184B5']
    high_lighted_2 = ['BT483']

    for i, cell_line in enumerate(_all_cell_lines):
        support_size = np.sum(np.logical_not(np.isnan(np.array(_means_accumulator[i]))))
        means_array = np.array(_means_accumulator[i])[support][argsorter]
        mean_fit = np.nanmean(means_array)
        g_coeff = gini_coeff(means_array)

        if support_size > 15:
            if cell_line in high_lighted_1:
                plt.plot([g_coeff], [mean_fit],
                         'o', color='r',
                         label=cell_line
                     # label='%s g: %.2f af: %.2f s: %s' % (cell_line, g_coeff, mean_fit, support_size)
                         )

            elif cell_line in high_lighted_2:
                plt.plot([g_coeff], [mean_fit],
                         'o',
                         label=cell_line
                     # label='%s g: %.2f af: %.2f s: %s' % (cell_line, g_coeff, mean_fit, support_size)
                         )

            else:
                plt.plot([g_coeff], [mean_fit],
                         'o', color='k',
                     # label='%s g: %.2f af: %.2f s: %s' % (cell_line, g_coeff, mean_fit, support_size)
                         )

    plt.xlabel('gini coefficient')
    plt.ylabel('average fitness across conditions')
    plt.legend(ncol=2)
    plt.show()

    _means_accumulator = np.array(_means_accumulator)
    baseline = _means_accumulator[ref_strain_idx, :][support][argsorter][np.newaxis, :]
    _norm_sorted_means = _means_accumulator[:, support][:, argsorter] / baseline
    _sorted_names = np.array(_names_accumulator)[support][argsorter]

    return _norm_sorted_means, _sorted_names


def plot_normalized(_norm_sorted_means, _sorted_names, _all_cell_lines):
    accumulator = []
    ramp = np.linspace(0, _sorted_names.shape[0], _sorted_names.shape[0]).tolist()
    cmap = mp.cm.get_cmap(name='Paired')

    ln = len(_all_cell_lines)
    for i, cell_line in enumerate(_all_cell_lines):
        support_size = np.sum(np.logical_not(np.isnan(np.array(_norm_sorted_means[i]))))

        if support_size < 10:
            continue

        else:
            log_transformed = np.log2(np.array(_norm_sorted_means[i]))
            accumulator.append(log_transformed)
            plt.plot(ramp,
                     log_transformed, '.',
                     label='%s' % cell_line,
                     color=cmap(i/float(ln)))

    accumulator = np.array(accumulator)
    means = np.nanmean(accumulator, axis=0)
    stds = np.nanstd(accumulator, axis=0)

    plt.plot(ramp, means, label='mean')
    plt.plot(ramp, stds, label='stds')

    mp.rc('font', size=10)
    plt.xticks(ramp, _sorted_names, rotation='vertical')
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

# ????
def cell_line_fingerprint(all_cell_lines_arr, means_accumulator, errs_accumulator, names_accumulator):
    means_accumulator, errs_accumulator, all_cell_lines_arr, names_accumulator = \
        supporting_functions.preformat(means_accumulator, errs_accumulator, all_cell_lines_arr, names_accumulator)


def drug_fingerprint(all_cell_lines_arr, means_accumulator, errs_accumulator, names_accumulator):
    means_accumulator, errs_accumulator, all_cell_lines_arr, names_accumulator = \
        supporting_functions.preformat(means_accumulator, errs_accumulator, all_cell_lines_arr, names_accumulator)


def drug_combination(_all_cell_lines_arr, _means_accumulator, _errs_accumulator, _names_accumulator):

    targets, fda_status = supporting_functions.read_drug_info(_names_accumulator)

    _names_accumulator = [name + ' - ' + targets[i] + ' - ' + fda_status[i]
                          for i, name in enumerate(_names_accumulator)]
    _means_accumulator, _errs_accumulator, _all_cell_lines_arr, _names_accumulator = \
        supporting_functions.preformat(_means_accumulator, _errs_accumulator, _all_cell_lines_arr, _names_accumulator)

    support_matrix = np.zeros((_names_accumulator.shape[0], _names_accumulator.shape[0], _means_accumulator.shape[0]))
    reverse_look_up_pad = np.zeros((_names_accumulator.shape[0], _names_accumulator.shape[0], 2))
    names_len = _names_accumulator.shape[0]

    for i in range(0, names_len):
        for j in range(i, names_len):
            support_matrix[i, j, :] = _means_accumulator[:, i] * _means_accumulator[:, j]
            support_matrix[j, i, :] = _means_accumulator[:, i] * _means_accumulator[:, j]
            reverse_look_up_pad[i, j, :] = np.array([i, j])
            reverse_look_up_pad[j, i, :] = np.array([i, j])

    all_cell_lines = _all_cell_lines_arr.tolist()
    idx1 = all_cell_lines.index('184A1')
    idx2 = all_cell_lines.index('184B5')
    false_filter = np.zeros((_all_cell_lines_arr.shape[0])).astype(np.bool)
    false_filter[idx1] = True
    false_filter[idx2] = True
    false_filter[-1] = True
    false_filter = np.logical_not(false_filter)

    norm_mean = support_matrix[:, :, -1].copy()
    support_matrix = support_matrix[:, :, false_filter] / norm_mean[:, :, np.newaxis]

    combined_effect = np.zeros((_names_accumulator.shape[0], _names_accumulator.shape[0]))
    for i in range(0, names_len):
        for j in range(i, names_len):
            comparable_support = np.logical_and(np.logical_not(np.isnan(support_matrix[i, i, :])),
                                                np.logical_not(np.isnan(support_matrix[j, j, :])))
            comparable_value = np.nanmean(support_matrix[i, j, comparable_support])
            comparable_value = comparable_value / np.min(np.array([
                                np.nanmean(support_matrix[i, i, comparable_support]),
                                np.nanmean(support_matrix[j, j, comparable_support])]))
            combined_effect[i, j] = comparable_value
            combined_effect[j, i] = comparable_value

    mean_effect = np.nanmean(support_matrix, axis=2)
    support_of_effect = np.sum(np.logical_not(np.isnan(support_matrix)), axis=2)
    combined_effect[support_of_effect < 15] = np.median(combined_effect)

    sorting_index = hierchical_clustering(combined_effect, _names_accumulator)

    combined_effect = combined_effect[sorting_index, :]
    combined_effect = combined_effect[:, sorting_index]

    reverse_look_up_pad = reverse_look_up_pad[sorting_index, :, :]
    reverse_look_up_pad = reverse_look_up_pad[:, sorting_index, :]

    _names_accumulator = _names_accumulator[sorting_index]

    combined_effect[combined_effect > 0.99] = np.nan
    # combined_effect[combined_effect < 0.1] = np.nan

    plt.imshow(combined_effect, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.xticks(np.linspace(0, _names_accumulator.shape[0]-1,
                           _names_accumulator.shape[0]), _names_accumulator, rotation='vertical')
    plt.yticks(np.linspace(0, _names_accumulator.shape[0]-1,
                           _names_accumulator.shape[0]), _names_accumulator)

    mp.rc('font', size=8)
    plt.subplots_adjust(bottom=0.3)
    plt.show()

    combined_effect[np.isnan(combined_effect)] = 0

    nz = np.nonzero(np.triu(combined_effect))
    acc_dict = {}

    for _i, _j in zip(nz[0], nz[1]):
        acc_dict[(_i, _j)] = combined_effect[_i, _j]

    sort_acc_dict = sorted(acc_dict.iteritems(), key=lambda x: x[1])

    print len(sort_acc_dict)

    for (_i, _j), val in sort_acc_dict[:10]:

        i, j = tuple(reverse_look_up_pad[_i, _j, :].tolist())

        plt.title('combination score: %.2f' % val)

        void_support = np.empty_like(_means_accumulator[:, 1])[:, np.newaxis]
        void_support.fill(np.nan)
        means_stack = np.hstack((_means_accumulator[:, [i, j]],
                                 void_support,
                                 (_means_accumulator[:, i] * _means_accumulator[:, j])[:, np.newaxis]))
        names_stack = _names_accumulator[[i, j]].tolist() + ['', 'combined']
        plt.imshow(means_stack, cmap='coolwarm', interpolation='nearest', vmin=0, vmax=2)
        plt.colorbar()
        plt.yticks(np.linspace(0, _all_cell_lines_arr.shape[0] - 1,
                               _all_cell_lines_arr.shape[0]), _all_cell_lines_arr)
        plt.xticks(np.linspace(0, 3, 4), names_stack, rotation='vertical')

        mp.rc('font', size=8)
        plt.subplots_adjust(bottom=0.3)
        plt.show()


if __name__ == '__main__':

    all_cell_lines, concs_effective_range = get_concentrations_of_interest(contracted_range=2,
                                                                           err_silencing=False)
    all_cell_lines_arr, means_accumulator,\
        errs_accumulator, names_accumulator = stack_data_in_range_of_interest(concs_effective_range)

    # drug_combination(all_cell_lines_arr, means_accumulator, errs_accumulator, names_accumulator)
    # means_accumulator = _95p_center(means_accumulator, errs_accumulator)
    norm_sorted_means, sorted_names = plot_response(means_accumulator,
                                                    errs_accumulator,
                                                    all_cell_lines_arr,
                                                    names_accumulator,
                                                    ref_strain='BT483',
                                                    # ref_strain='WT_proxy',
                                                    normalize=False,
                                                    log=False,
                                                    sort_by='ref_strain')

    plot_normalized(norm_sorted_means, sorted_names, all_cell_lines)
