import numpy as np
from matplotlib import pyplot as plt
from chiffatools.linalg_routines import rm_nans
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
            nan_mask = np.logical_not(np.isnan(full_values[i, :, j]))
            plt.errorbar(temp_concs[nan_mask], full_values[i, nan_mask, j],
                         yerr=[errs[0][i, nan_mask, j], errs[1][i, nan_mask, j]], fmt='.', color=color, alpha=0.25)
            plt.errorbar(temp_concs[nan_mask], values[i, nan_mask, j],
                         yerr=[errs[0][i, nan_mask, j], errs[1][i, nan_mask, j]], fmt='.', color=color)



def summary_plot(means, mean_err, concentrations, anchor, color='black', legend='', nofill=False):

    # TODO: inject nan to mark that the control is different from the main sequence.

    ax = plt.subplot(111)
    ax.set_xscale('log')

    nanmask = np.logical_not(np.isnan(means))
    if not np.all(np.logical_not(nanmask)):
        concentrations[0] = anchor
        plt.errorbar(concentrations[nanmask], means[nanmask], yerr=mean_err[nanmask], color=color, label=legend)
        ymax = means[nanmask] + mean_err[nanmask]
        ymin = means[nanmask] - mean_err[nanmask]
        if not nofill:
            plt.fill_between(concentrations[nanmask], ymax, ymin, facecolor=color, alpha=0.25)


def vector_summary_plot(means_array, error_array, concentrations_array, anchor, legend_array=None, color='black'):
    if legend_array is None:
        legend_array = np.zeros_like(means_array[:, 0])

    for i in range(0, means_array.shape[0]):
        nanmask = np.logical_not(np.isnan(means_array[i, :]))
        if not np.all(np.logical_not(nanmask)):
            summary_plot(means_array[i, nanmask], error_array[i, nanmask], concentrations_array[i, nanmask], anchor, color, legend_array[i])


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