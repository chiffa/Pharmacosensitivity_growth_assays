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
from slugify import slugify
from scipy.linalg import block_diag
from multiprocessing import Process

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


    def retrieve(self, cell, drug, correct_background=True, correct_lower_boundary=True):

        drug_c_array = np.array([0]+[2**_i for _i in range(0, 9)])*0.5**8

        def nan(_drug_n):
            return np.all(np.isnan(self.storage[cell_n, _drug_n]))

        def helper_round(T_container):
            T_container_vals = [np.repeat(T_container[cell_n, drug_n][:, np.newaxis], 10, axis=1) for drug_n in drugs_nos]
            T_container_vals = SF.block_fusion(T_container_vals)
            T_container_vals = T_container_vals[:, c_argsort]
            return T_container_vals

        cell_n = self.cell_idx[cell]
        retained_drugs = [drug_v for drug_v in self.drug_versions[drug] if not nan(self.drug_idx[drug_v])]
        if retained_drugs == []:
            retained_drugs = [drug_v for drug_v in self.drug_versions[drug]]

        drugs_nos = [self.drug_idx[drug_v] for drug_v in retained_drugs]
        drug_vals = [self.storage[cell_n, drug_n].copy() for drug_n in drugs_nos]

        drug_c = [drug_v[1]*drug_c_array for drug_v in retained_drugs]
        anchor = np.min(np.array([drug_v[1] for drug_v in retained_drugs]))
        anchor = anchor/2.**10
        drug_vals = SF.block_fusion(drug_vals)

        drug_c = np.hstack(drug_c)
        c_argsort = np.argsort(drug_c)

        drug_c = drug_c[c_argsort]
        drug_vals = drug_vals[:, c_argsort, :]

        T0_bck_vals = helper_round(self.T0_background.copy())
        TF_bck_vals = helper_round(self.TF_background.copy())
        T0_vals = helper_round(self.T0_median.copy())
        noize_dispersion = [helper_round(self.background[:, :, :, i].copy()) for i in range(0,4)]
        noize_dispersion = SF.estimate_differences(noize_dispersion)

        if correct_background:
            drug_vals -= TF_bck_vals[:,:, np.newaxis]
            T0_vals -= T0_bck_vals[:, :]

        if correct_lower_boundary:
            drug_vals = SF.get_boundary_correction(drug_vals, self.std_of_tools)

        # print 'drug_vals', drug_vals
        # print 'drug_c', drug_c
        # print 'T0_vals', T0_vals

        return drug_vals, drug_c, T0_vals, noize_dispersion, anchor

    def run_QC(self):
        pass

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

    def run_QC(self):
        pass


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

        self.cassificant_index = cell_idx                   # celline to index
        self.header = header                                # drug to index
        self.classificant_index_rv = cell_idx_rv            # index to cell_line
        self.markers = markers


def fragmented_round(cell_line, drug, color='black', plot_type = 10, injected_anchor=None):

    render_type = int(str(plot_type)[0])
    normalization_type = int(str(plot_type)[1])

    plot_touched = False

    TF_OD, concentrations, T0_median, noize_dispersion, anchor = hr.retrieve(cell_line, drug)
    if injected_anchor is not None:
        anchor = injected_anchor
    clean_mask = SF.clean_nans(TF_OD)

    # TODO: proper anchor management. Problem: we need to know the anchors of all the three elements before we are
    # really able to do any drawing.

    if not np.any(clean_mask):
        return plot_touched

    TF_OD, T0_median = (TF_OD[clean_mask, :, :], T0_median[clean_mask, :])
    TF_corrected, means_arr, errs_arr, unique_concs_stack = SF.correct_plates(TF_OD, concentrations, hr.std_of_tools)

    if render_type == 1:
        plot_touched = True
        PD.raw_plot(TF_corrected, TF_OD, concentrations, hr.std_of_tools, color=color)
        PD.vector_summary_plot(means_arr, errs_arr, unique_concs_stack, anchor, color=color)
        # plt.show()

    clean_mask = SF.clean_nans(TF_corrected)

    if not np.any(clean_mask):
        return plot_touched

    TF_corrected, T0_corrected = (TF_corrected[clean_mask, :, :], T0_median[clean_mask, :])
    means_arr, errs_arr = (means_arr[clean_mask, :], errs_arr[clean_mask, :])

    norm_factor = None

    if normalization_type == 0:
        norm_factor = SF.retrieve_normalization_factor(means_arr)

    if normalization_type == 1:
        norm_factor = SF.retrieve_normalization_factor(T0_corrected)

    norm_plate, norm_means, norm_errs, norm_std_of_tools = SF.normalize(TF_corrected, means_arr,
                                                                        errs_arr, hr.std_of_tools,
                                                                        norm_factor)

    if render_type == 2:
        plot_touched = True
        PD.vector_summary_plot(norm_means, norm_errs, unique_concs_stack, anchor, color=color)
        # plt.show()

    means, errs, unique_concs = SF.combine(norm_plate, concentrations, np.max(norm_std_of_tools))

    if render_type == 3:
        plot_touched = True
        PD.summary_plot(means, errs, unique_concs, anchor, color = color)
        # plt.show()

    return plot_touched


def compare_to_htert(cell_line, drug, standardized, plot_type=1):
    plt.title('%s, %s' % (cell_line, drug))

    print '{0:15}'.format(cell_line),
    print '{0:20}'.format(drug),
    print '\t',

    _, _, _, _, r_anchor = hr.retrieve('184A1', drug)
    _, _, _, _, g_anchor = hr.retrieve('184A1', drug)
    _, _, _, _, b_anchor = hr.retrieve('184A1', drug)

    anchor = np.nanmin(np.array([r_anchor, g_anchor, b_anchor]))

    rt = fragmented_round('184A1', drug, 'red', plot_type, anchor)
    print 'rt',
    gt = fragmented_round('184B5', drug, 'green', plot_type, anchor)
    print 'gt',
    bt = fragmented_round(cell_line, drug, 'black', plot_type, anchor)
    print 'bt',

    if not bt:
        print '-'
        return ''

    plt.gcf().set_size_inches(25, 15, forward=True)
    plt.autoscale(tight=True)
    plt.legend()

    cell_line = slugify(cell_line)
    drug = slugify(drug)


    SF.safe_dir_create('../analysis_runs/%s/vrac/'%SF.type_map(plot_type))
    plt.savefig('../analysis_runs/%s/vrac/%s - %s.png'%(SF.type_map(plot_type), cell_line, drug) )

    SF.safe_dir_create('../analysis_runs/%s/by_drug/%s/'%(SF.type_map(plot_type), drug))
    plt.savefig('../analysis_runs/%s/by_drug/%s/%s.png'%(SF.type_map(plot_type), drug, cell_line))

    SF.safe_dir_create('../analysis_runs/%s/by_cell_line/%s/'%(SF.type_map(plot_type), cell_line))
    plt.savefig('../analysis_runs/%s/by_cell_line/%s/%s.png'%(SF.type_map(plot_type), cell_line, drug))

    print '+'
    return ''


def loop(plot_type=10):

    def nan(_drug_n):
            return np.all(np.isnan(hr.storage[cell_n, _drug_n]))

    print 'starting loop with the following parameters: %s' % (plot_type)

    for drug in hr.drug_versions.keys():
        for cell_line, cell_n in hr.cell_idx.iteritems():
            if not cell_line in ['184A1', '184B5']:
                if [drug_v for drug_v in hr.drug_versions[drug] if not nan(hr.drug_idx[drug_v])]:
                    compare_to_htert(cell_line, drug, True, plot_type)
                    plt.clf()


def test_GI_50_reader():
    tr = GI_50_reader('C:\\Users\\Andrei\\Desktop', 'sd02.tsv')
    print tr.retrieve('HCC1954', 'XRP44X')


if __name__ == "__main__":
    hr = raw_data_reader('C:\\Users\\Andrei\\Desktop', 'gb-breast_cancer.tsv')
    tr = GI_50_reader('C:\\Users\\Andrei\\Desktop', 'sd05-bis.tsv')

    print fragmented_round('600MPE', 'GSK1838705', plot_type=30)
    plt.show()

    # loop(20)