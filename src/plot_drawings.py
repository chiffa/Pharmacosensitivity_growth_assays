__author__ = 'Andrei'

import numpy as np
from matplotlib import pyplot as plt
from chiffatools.Linalg_routines import rm_nans
from chiffatools.dataviz import better2D_desisty_plot
import supporting_functions as SF
from scipy import stats


def quick_hist(data):
    plt.hist(np.log10(rm_nans(data)), bins=20)
    plt.show()


def show_2d_array(data):
    plt.imshow(data, interpolation='nearest', cmap='coolwarm')
    plt.colorbar()
    plt.show()


def correlation_plot(x, y):
    plt.plot(x, y, '.k')
    plt.show()
    better2D_desisty_plot(x, y)
    plt.show()
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    print "r-squared:", r_value**2


def raw_plot(values, full_values, concentrations, noise_level, color):

    m_i = values.shape[0]
    m_j = values.shape[2]

    ax = plt.subplot(111)

    ax.set_xscale('log')

    msk = concentrations == 0.0
    concentrations[msk] = np.min(concentrations[np.logical_not(msk)])/4

    if type(noise_level) == np.float64 or type(noise_level) == float:
        errs = np.empty_like(values)
        errs.fill(noise_level)
        errs = [errs, errs]

    if type(noise_level) == np.ndarray:
        errs = [noise_level, noise_level]

    if type(noise_level) == tuple:
        errs = [noise_level[0], noise_level[1]]

    for i in range(0, m_i):
        for j in range(0, m_j):
            # temp_concs = concentrations
            temp_concs = concentrations*np.random.uniform(0.95, 1.05, 1)
            plt.errorbar(temp_concs, full_values[i, :, j], yerr=[errs[0][i, :, j], errs[1][i, :, j]], fmt='.', color=color, alpha=0.25)
            plt.errorbar(temp_concs, values[i, :, j], yerr=[errs[0][i, :, j], errs[1][i, :, j]], fmt='.', color=color)



def summary_plot(means, mean_err, concentrations, anchor=None, color='black', legend=''):
    if anchor is None:
        concentrations[0] = concentrations[1]/4
    else:
        concentrations[0] = anchor
    plt.errorbar(concentrations, means, yerr=mean_err, color=color, label=legend)
    ymax = means + mean_err
    ymin = means - mean_err
    plt.fill_between(concentrations, ymax, ymin, facecolor=color, alpha=0.25)


def bi_plot(raw_points, full_raw_points, concentrations, std_of_tools, filter_level=None, GI_50=np.nan, color='black', legend=''):
    rp_noise = stats.norm.interval(.95, scale=std_of_tools)[1]
    means, errs, stds, freedom_degs, unique_concs = SF.compute_stats(raw_points, concentrations, std_of_tools)
    anchor = None
    if filter_level is not None:
        msk = errs < filter_level*std_of_tools
        means = means[msk]
        errs = errs[msk]
        anchor = unique_concs[1]/4
        unique_concs = unique_concs[msk]
        for i, conc in enumerate(concentrations):
            if conc not in unique_concs:
                raw_points[:, i, :] = np.nan
    raw_plot(raw_points, full_raw_points, concentrations, rp_noise, color)
    summary_plot(means, errs, unique_concs, anchor, color, legend=legend)
    if GI_50 != np.nan:
        plt.axvline(GI_50, color=color)

    return means, errs, unique_concs


def pretty_gradual_plot(data, concentrations, strain_name_map, drug_name, blank_line=200):

    def inner_scatter_plot(mean, std, relative, limiter=4):
        series = np.zeros(mean.shape)
        cell_type = np.zeros(mean.shape)
        for i, name in enumerate(names):
            series[i, :] = np.arange(i, c.shape[0]*(len(names)+40)+i, len(names)+40)
            cell_type[i, :] = i
            plt.scatter(series[i, :], mean[i, :], c=cm(i/float(len(names))), s=35, label=name)
        plt.errorbar(series.flatten(), mean.flatten(), yerr=std.flatten(), fmt=None, capsize=0)
        plt.xticks(np.mean(series, axis=0), c)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=len(names)/limiter, mode="expand", borderaxespad=0.,prop={'size':6})
        if not relative:
            plt.axhline(y=blank_line)
        plt.show()

    filter = np.all(np.logical_not(np.isnan(data)), axis=(1, 2))
    names = [strain_name_map[i] for i in filter.nonzero()[0].tolist()]
    c = concentrations[filter, :][0, :]
    mean = np.nanmean(data[filter, :, :], axis=-1)
    std = np.nanstd(data[filter, :, :], axis=-1)
    cm = plt.cm.get_cmap('spectral')

    refmean = mean[:, 0].reshape((mean.shape[0], 1))
    refstd = std[:, 0].reshape((mean.shape[0], 1))
    rel_mean, rel_std = (mean/refmean, np.sqrt(np.power(refstd, 2)+np.power(std, 2))/mean)

    inner_scatter_plot(mean, std, False)
    inner_scatter_plot(rel_mean, rel_std, True)

    mean_mean = np.nanmean(mean, axis=0)
    std_mean = np.nanstd(mean, axis=0)
    mean_std = np.nanmean(std, axis=0)
    total_std = np.sqrt(np.power(std_mean, 2) + np.power(mean_std, 2))
    confusables = np.sum(mean - std < blank_line, axis=0) / float(len(names))

    rel_mean_mean = np.nanmean(rel_mean, axis=0)
    rel_std_mean = np.nanstd(rel_mean, axis=0)
    rel_mean_std = np.nanmean(rel_std, axis=0)
    rel_total_std = np.sqrt(np.power(rel_std_mean, 2) + np.power(rel_mean_std, 2))

    plt.subplot(212)
    plt.plot(mean_mean, c=cm(0.00), label='mean of mean')
    plt.plot(mean_std, c=cm(.25), label='mean of std')
    plt.plot(std_mean, c=cm(.50), label='std of mean')
    plt.plot(total_std, c=cm(0.75), label='total std')
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0.,prop={'size':8})
    plt.axhline(y=blank_line)

    plt.subplot(211)
    plt.plot(rel_mean_mean, c=cm(0.00), label='mean of mean')
    plt.plot(rel_mean_std, c=cm(.25), label='mean of std')
    plt.plot(rel_std_mean, c=cm(.50), label='std of mean')
    plt.plot(rel_total_std, c=cm(0.75), label='total std')
    plt.plot(confusables, c=cm(0.9), label='confusable with null')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0.,prop={'size':8})

    plt.show()