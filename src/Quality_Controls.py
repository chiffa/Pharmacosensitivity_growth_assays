__author__ = 'Andrei'

import numpy as np
from supporting_functions import p_stabilize
from plot_drawings import show_2d_array, quick_hist, correlation_plot
from chiffatools.dataviz import smooth_histogram
from scipy.stats import norm

# TODO: cell_drug_replicates analysis
# TODO: plate_specific noise level : correlation between T0 and TF difference between blank wells
# TODO: background (T0+TF) histogram (different colors + transparency) (use smooth histogram from chiffatools/dataviz)
# TODO: compute violating plates ( at least one blank well, at T = 0 or T = T_final have an OD > 1000


def check_replicates(cl_drug_replicates):
    print np.sum(np.isnan(cl_drug_replicates).astype(np.int))
    print np.nanmean(cl_drug_replicates)
    print np.nanmin(cl_drug_replicates)
    print np.nanmax(cl_drug_replicates)
    print np.nansum(cl_drug_replicates)
    print 1, np.sum((cl_drug_replicates == 1).astype(np.int))
    print 2, np.sum((cl_drug_replicates == 2).astype(np.int))
    print 3, np.sum((cl_drug_replicates == 3).astype(np.int))
    print 4, np.sum((cl_drug_replicates == 4).astype(np.int))
    print 5, np.sum((cl_drug_replicates == 5).astype(np.int))
    print 6, np.sum((cl_drug_replicates == 6).astype(np.int))
    print 7, np.sum((cl_drug_replicates == 7).astype(np.int))
    print 8, np.sum((cl_drug_replicates == 8).astype(np.int))
    print 'sup', np.sum((cl_drug_replicates > 6).astype(np.int))
    show_2d_array(cl_drug_replicates)


def check_blanks(background, background_noise):
    quick_hist(background)
    background_noise = p_stabilize(background_noise, 0.5)
    correlation_plot(background_noise[:, 0], background_noise[:, 1])
    quick_hist(np.nanmean(p_stabilize(background_noise, 0.5), 1))


def check_background(background):
    """
    checks that background actually follows poisson distribution

    :param background:
    :return:
    """
    background = p_stabilize(background, 0.5)
    mean = np.nanmean(background)
    std = np.nanstd(background)
    smooth_histogram(background)
    smooth_histogram(norm.rvs(mean, std, size=1000),'g')