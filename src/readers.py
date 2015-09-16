__author__ = 'Andrei'


from csv import reader
from os import path
from pprint import pprint
import numpy as np
from chiffatools.Linalg_routines import rm_nans
import supporting_functions as SF
import plot_drawings as PD
from plot_drawings import quick_hist, show_2d_array, correlation_plot, raw_plot, summary_plot
from collections import defaultdict
import Quality_Controls as QC
from matplotlib import pyplot as plt
from StringIO import StringIO

class raw_data_reader(object):

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

        cell_idx = SF.index(set(cells))
        drug_idx = SF.index(set(drugs))
        plates_idx = SF.index(set(plates))
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
            SF.broadcast(test_array[6:36])
            for row in rdr:
                cell_no = cell_idx[row[0]]
                drug_no = drug_idx[(row[1], float(row[47]))]
                plate_no = plates_idx[row[2]]
                depth_index = min(cl_drug_replicates[cell_no, drug_no], depth_limiter-1)
                storage[cell_no, drug_no, depth_index, :, :] = SF.broadcast(row[6:36])
                background[cell_no, drug_no, depth_index, :] = SF.lgi(row, [4, 5, 36, 37])
                T0_median[cell_no, drug_no, depth_index] = row[38]
                T0_background[cell_no, drug_no, depth_index] = np.mean(SF.lgi(row, [4, 5]).astype(np.float64)).tolist()
                TF_background[cell_no, drug_no, depth_index] = np.mean(SF.lgi(row, [36, 37]).astype(np.float64)).tolist()
                background_noise[plate_no, :] = np.abs(SF.lgi(row, [4, 36]).astype(np.float64) - SF.lgi(row, [5, 37]).astype(np.float64))
                cl_drug_replicates[cell_no, drug_no] += 1

        cl_drug_replicates[cl_drug_replicates < 1] = np.nan

        alpha_bound = np.percentile(rm_nans(background_noise), 100 - alpha_bound_percentile)
        std_of_tools = np.percentile(rm_nans(background_noise), 66)

        background = SF.p_stabilize(background, 0.5)
        T0_background = SF.p_stabilize(T0_background, 0.5)
        TF_background = SF.p_stabilize(TF_background, 0.5)

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


    def retrieve(self, cell, drug, correct_plates=True, correct_background=True):

        drug_c_array = np.array([0]+[2**_i for _i in range(0, 9)])*0.5**8

        def nan(_drug_n):
            return np.all(np.isnan(self.storage[cell_n, _drug_n]))

        def helper_round(T_container):
            T_container_vals = [np.repeat(T_container[cell_n, drug_n][:, np.newaxis], 10, axis=1) for drug_n in drugs_nos]
            T_container_vals = np.hstack(T_container_vals)
            T_container_vals = T_container_vals[:, c_argsort]
            return T_container_vals

        def plate_error_correction(value_set):
            for i in range(0, value_set.shape[0]):
                mean_arr = np.nanmean(value_set[i, :, :], axis=1)
                if np.max(mean_arr) - np.min(mean_arr) < 10*self.std_of_tools:
                    print "plate-wide lack of action detected for %s, %s at level %s" % (cell, drug, i)
                    print mean_arr, np.max(mean_arr)-np.min(mean_arr)
                    value_set[i, :, :] = np.nan
            return value_set

        cell_n = self.cell_idx[cell]
        retained_drugs = [drug_v for drug_v in self.drug_versions[drug] if not nan(self.drug_idx[drug_v])]
        print retained_drugs

        drugs_nos = [self.drug_idx[drug_v] for drug_v in retained_drugs]
        drug_vals = [self.storage[cell_n, drug_n] for drug_n in drugs_nos]
        if correct_plates:
            drug_vals = [plate_error_correction(val) for val in drug_vals]
        drug_c = [drug_v[1]*drug_c_array for drug_v in retained_drugs]

        drug_vals = np.hstack(drug_vals)
        drug_c = np.hstack(drug_c)

        c_argsort = np.argsort(drug_c)

        drug_c = drug_c[c_argsort]
        drug_vals = drug_vals[:, c_argsort, :]  # standard error of mean is the standard deviation divided by the sqrt of number of non-nul elements

        T0_bck_vals = helper_round(self.T0_background)
        TF_bck_vals = helper_round(self.TF_background)
        T0_vals = helper_round(self.T0_median)



        return drug_vals, drug_c, T0_bck_vals, TF_bck_vals, T0_vals


class GI_50_reader(object):

    def __init__(self, pth, fle):

        cells = []
        data_matrix = []
        with open(path.join(pth, fle)) as src:
            rdr = reader(src, dialect='excel-tab')
            rdr.next()
            rdr.next()
            drugs = rdr.next()[1:]
            for row in rdr:
                cells.append(row[0])
                data_matrix.append(np.genfromtxt(np.array(row[1:])).astype(np.float64))

        cell_idx = SF.index(cells)
        drug_idx = SF.index(drugs)

        cell_idx_rv = dict([(value, key) for key, value in cell_idx.iteritems()])
        drug_idx_rv = dict([(value, key) for key, value in drug_idx.iteritems()])

        self.cell_idx = cell_idx                # celline to index
        self.drug_idx = drug_idx                # drug to index
        self.cell_idx_rv = cell_idx_rv          # index to cell_line
        self.drug_idx_rv = drug_idx_rv
        self.GI_50 = np.array(data_matrix)


    def retrieve(self, celline, drug):
        if self.cell_idx.has_key(celline) and self.drug_idx.has_key(drug):
            return self.GI_50[self.cell_idx[celline], self.drug_idx[drug]]
        else:
            return np.NaN


class classification_reader(object):

    def __init__(self, pth, fle):
        cells = []
        markers = []
        with open(path.join(pth, fle)) as src:
            rdr = reader(src, dialect='excel-tab')
            rdr.next()
            rdr.next()
            header = rdr.next()[1:]
            for row in rdr:
                cells.append(row[0])
                markers.append(row[1:])

        cell_idx = SF.index(cells)

        cell_idx_rv = dict([(value, key) for key, value in cell_idx.iteritems()])

        self.cassificant_index = cell_idx                # celline to index
        self.header = header                    # drug to index
        self.classificant_index_rv = cell_idx_rv          # index to cell_line
        self.markers = markers


def test_raw_data_reader():
    hr = raw_data_reader('C:\\Users\\Andrei\\Desktop', 'gb-breast_cancer.tsv')
    tr = GI_50_reader('C:\\Users\\Andrei\\Desktop', 'sd05-bis.tsv')
    # cr = classification_reader('C:\\Users\\Andrei\\Desktop', 'sd01-bis.tsv')
    # dr = classification_reader('C:\\Users\\Andrei\\Desktop', 's5.tsv')

    # TF_OD, concentrations, T0_bck, TF_bck, T0_median = hr.retrieve('HCC202', 'Rapamycin')
    # GI_50 = 10**(-tr.retrieve('HCC202', 'Rapamycin'))

    TF_OD, concentrations, T0_bck, TF_bck, T0_median = hr.retrieve('184A1', '17-AAG')
    GI_50 = 10**(-tr.retrieve('184A1', '17-AAG'))

    # QC.check_reader_consistency([hr.cell_idx.keys(), tr.cell_idx.keys(), cr.cassificant_index.keys()])
    # QC.check_reader_consistency([hr.drug_versions.keys(), tr.drug_idx.keys(), dr.cassificant_index.keys()])

    T0, TF, fold_growth, sigmas = SF.correct_values(TF_OD, T0_bck, TF_bck, T0_median, hr.std_of_tools)

    PD.bi_plot(fold_growth, concentrations, hr.std_of_tools)
    PD.bi_plot(sigmas, concentrations, std_of_tools=1., filter_level=9., GI_50=GI_50)


def test_GI_50_reader():
    tr = GI_50_reader('C:\\Users\\Andrei\\Desktop', 'sd02.tsv')
    print tr.retrieve('HCC1954', 'XRP44X')


if __name__ == "__main__":
    test_raw_data_reader()
    # test_GI_50_reader()

    #TODO: plot raw points in black for points that were kept and in red - points that were eliminated