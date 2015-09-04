__author__ = 'Andrei'


from csv import reader
from os import path
from pprint import pprint
import numpy as np
from chiffatools.Linalg_routines import rm_nans
from supporting_functions import index, broadcast, lgi, p_stabilize, extract
from plot_drawings import quick_hist, show_2d_array, correlation_plot, raw_plot
from collections import defaultdict
import Quality_Controls as QC

class historical_reader(object):

    def __init__(self, pth, fle, alpha_bound_percentile=2):

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
        noise_level = np.percentile(rm_nans(background_noise), 66)

        T0_background = p_stabilize(T0_background, 0.5)
        TF_background = p_stabilize(TF_background, 0.5)

        storage_dblanc = storage - TF_background[:,:,:, np.newaxis, np.newaxis]
        T0_median_dblanc = T0_median - T0_background

        self.header = header                    # header line
        self.cell_idx = cell_idx                # celline to index
        self.drug_idx = drug_idx                # drug to index
        self.cell_idx_rv = cell_idx_rv          # index to cell_line
        self.drug_idx_rv = drug_idx_rv          # index to drug
        self.storage = storage_dblanc           # for each cell_line, drug, concentration, contains the three replicates
        self.background = background            # for each cell_line and drug contains T0_1, T0_2 and T_final, T_final backgrounds
        self.T0_median = T0_median_dblanc       # for each cell_line and drug contains T0
        self.alpha_bound = alpha_bound          # value below which difference is not considered as significant.
        self.noise_level = noise_level          # equivalent of the variance for a centered normal distribution (66% encompassing bound )
        self.background_noise = background_noise
        self.drug_versions = dict(drug_versions)      # contains the associated drug-concentration pairs for every unique drug


    def return_relevant_values(self):
        render_dict = {}
        for property, value in vars(self).iteritems():
            render_dict[str(property)] = value
        return render_dict


if __name__ == "__main__":
    hr = historical_reader('C:\\Users\\Andrei\\Desktop', 'gb-breast_cancer.tsv')
    pprint(hr.return_relevant_values().keys())
    extr_vals, extr_concs = extract(hr.storage, 'HCC1419', 'PD98059', hr.drug_versions, hr.cell_idx, hr.drug_idx)
    # use reverse lookup to find an element with a lot of replications
    raw_plot(extr_vals, extr_concs)