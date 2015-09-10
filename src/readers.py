__author__ = 'Andrei'


from csv import reader
from os import path
from pprint import pprint
import numpy as np
from chiffatools.Linalg_routines import rm_nans
from supporting_functions import index, broadcast, lgi, p_stabilize, extract, compute_stats, correct_values
import supporting_functions as SF
from plot_drawings import quick_hist, show_2d_array, correlation_plot, raw_plot, summary_plot
from collections import defaultdict
import Quality_Controls as QC
from matplotlib import pyplot as plt

class historical_reader(object):

    def __init__(self, pth, fle, alpha_bound_percentile=5):

        cells = []
        drugs = []
        drug_versions = defaultdict(list)
        plates =[]
        with open(path.join(pth, fle)) as src:
            rdr = reader(src, dialect='excel-tab')
            header = rdr.next()
            for row in rdr:
                expanded_drug_name = (row[1], float(row[47]))
                cells.append(row[0])
                drug_versions[row[1]].append(expanded_drug_name)
                drugs.append(expanded_drug_name)
                plates.append(row[2])

        cell_idx = index(set(cells))
        drug_idx = index(set(drugs))
        plates_idx = index(set(plates))
        drug_versions = dict([(key, list(set(values))) for key, values in drug_versions.iteritems()])

        cell_idx_rv = dict([(value, key) for key, value in cell_idx.iteritems()])
        drug_idx_rv = dict([(value, key) for key, value in drug_idx.iteritems()])
        plates_idx_rv = dict([(value, key) for key, value in plates_idx.iteritems()])

        cellno = len(cell_idx)
        drugno = len(drug_idx)
        platesno = len(plates_idx)

        depth_limiter = 7

        storage = np.empty((cellno, drugno, depth_limiter, 10, 3))
        storage.fill(np.NaN)

        background = np.empty((cellno, drugno, depth_limiter, 4))
        background.fill(np.NaN)

        T0_median = np.empty((cellno, drugno, depth_limiter))
        T0_median.fill(np.NaN)

        T0_background = np.empty((cellno, drugno, depth_limiter))
        T0_background.fill(np.NaN)

        TF_background = np.empty((cellno, drugno, depth_limiter))
        TF_background.fill(np.NaN)

        background_noise = np.empty((platesno, 2))
        background_noise.fill(np.NaN)

        cl_drug_replicates = np.zeros((cellno, drugno))

        with open(path.join(pth, fle)) as src:
            rdr = reader(src, dialect='excel-tab')
            test_array = rdr.next()
            broadcast(test_array[6:36])
            for row in rdr:
                cell_no = cell_idx[row[0]]
                drug_no = drug_idx[(row[1], float(row[47]))]
                plate_no = plates_idx[row[2]]
                depth_index = min(cl_drug_replicates[cell_no, drug_no], depth_limiter-1)
                storage[cell_no, drug_no, depth_index, :, :] = broadcast(row[6:36])
                background[cell_no, drug_no, depth_index, :] = lgi(row, [4, 5, 36, 37])
                T0_median[cell_no, drug_no, depth_index] = row[38]
                T0_background[cell_no, drug_no, depth_index] = np.mean(lgi(row, [4, 5]).astype(np.float64)).tolist()
                TF_background[cell_no, drug_no, depth_index] = np.mean(lgi(row, [36, 37]).astype(np.float64)).tolist()
                background_noise[plate_no, :] = np.abs(lgi(row, [4, 36]).astype(np.float64) - lgi(row, [5, 37]).astype(np.float64))
                cl_drug_replicates[cell_no, drug_no] += 1

        cl_drug_replicates[cl_drug_replicates < 1] = np.nan

        alpha_bound = np.percentile(rm_nans(background_noise), 100 - alpha_bound_percentile)
        std_of_tools = np.percentile(rm_nans(background_noise), 66)

        background = p_stabilize(background, 0.5)
        T0_background = p_stabilize(T0_background, 0.5)
        TF_background = p_stabilize(TF_background, 0.5)

        storage_dblanc = storage - TF_background[:,:,:, np.newaxis, np.newaxis]

        self.header = header                    # header line
        self.cell_idx = cell_idx                # celline to index
        self.drug_idx = drug_idx                # drug to index
        self.cell_idx_rv = cell_idx_rv          # index to cell_line
        self.drug_idx_rv = drug_idx_rv          # index to drug
        self.storage = storage_dblanc           # for each cell_line, drug, concentration, contains the three replicates
        self.background = background            # for each cell_line and drug contains T0_1, T0_2 and T_final, T_final backgrounds
        self.T0_background = T0_background
        self.TF_background = TF_background
        self.T0_median = T0_median              # for each cell_line and drug contains T0
        self.alpha_bound = alpha_bound          # value below which difference is not considered as significant.
        self.std_of_tools = std_of_tools          # equivalent of the variance for a centered normal distribution (66% encompassing bound )
        self.background_noise = background_noise
        self.drug_versions = dict(drug_versions)      # contains the associated drug-concentration pairs for every unique drug
        self.cl_drug_replicates = cl_drug_replicates


    def return_relevant_values(self):
        render_dict = {}
        for property, value in vars(self).iteritems():
            render_dict[str(property)] = value
        return render_dict


if __name__ == "__main__":
    hr = historical_reader('C:\\Users\\Andrei\\Desktop', 'gb-breast_cancer.tsv')
    pprint(hr.return_relevant_values().keys())
    # show_2d_array(hr.cl_drug_replicates)
    # print hr.cell_idx_rv[37], hr.drug_idx_rv[32]
    # print hr.std_of_tools

    # extr_vals, extr_concs, T0_bck, TF_bck, T0_median = extract(hr.storage, 'HCC2185', '17-AAG', hr.drug_versions,
    #                                                             hr.cell_idx, hr.drug_idx, hr.T0_background,
    #                                                             hr.TF_background, hr.T0_median)

    extr_vals, extr_concs, T0_bck, TF_bck, T0_median = extract(hr.storage, 'HCC202', 'Rapamycin', hr.drug_versions,
                                                                hr.cell_idx, hr.drug_idx, hr.T0_background,
                                                                hr.TF_background, hr.T0_median)

    # use reverse lookup to find an element with a lot of replications

    T0, TF, fold_growth, sigmas = correct_values(extr_vals, T0_bck, TF_bck, T0_median, hr.std_of_tools)

    raw_plot(fold_growth, extr_concs, hr.std_of_tools)
    means, errs, stds, freedom_degs, unique_concs = compute_stats(fold_growth, extr_concs, hr.std_of_tools)
    summary_plot(means, errs, unique_concs)
    plt.show()

    # TF_means, TF_errs, TF_stds, TF_dfs, unique_concs = compute_stats(TF, extr_concs, hr.noise_level)
    # T0_means, T0_errs, T0_stds, T0_dfs, unique_concs = compute_stats(T0[:, :, np.newaxis], extr_concs, hr.noise_level)

    alphas = SF.logistic_regression(TF, T0, extr_concs, hr.std_of_tools)

    raw_plot(alphas, extr_concs, 0.0)
    # # means, errs, unique_concs = compute_stats(extr_vals, extr_concs, hr.noise_level)
    # summary_plot(alpha_means, [np.sqrt(alpha_stds**2+alpha_mins**2)/np.sqrt(alpha_dfs),
    #                            np.sqrt(alpha_stds**2+alpha_maxs**2)/np.sqrt(alpha_dfs)], unique_concs)
    plt.show()

    # TODO: replace the upper bound in case the actual data point is below 0
