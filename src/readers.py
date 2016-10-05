from csv import reader
from os import path
from pprint import pprint
import numpy as np
from chiffatools.linalg_routines import rm_nans
import supporting_functions
import plot_drawings
from collections import defaultdict
import Quality_Controls as QC
from matplotlib import pyplot as plt
from StringIO import StringIO
from slugify import slugify
from scipy.linalg import block_diag
from multiprocessing import Process
from pickle import dump


class RawDataReader(object):

    def __init__(self, pth, fle, alpha_bound_percentile=5):

        cells = []
        drugs = []
        drug_versions = defaultdict(list)
        plates = []
        with open(path.join(pth, fle)) as src:
            rdr = reader(src, dialect='excel-tab')
            header = rdr.next()
            for row in rdr:
                expanded_drug_name = (row[1], float(row[47]))
                cells.append(row[0])
                drug_versions[row[1]].append(expanded_drug_name)
                drugs.append(expanded_drug_name)
                plates.append(row[2])

        cell_idx = supporting_functions.index(set(cells))
        drug_idx = supporting_functions.index(set(drugs))
        plates_idx = supporting_functions.index(set(plates))
        drug_versions = dict([(key, list(set(values)))
                              for key, values in drug_versions.iteritems()])

        cell_idx_rv = dict([(value, key) for key, value in cell_idx.iteritems()])
        drug_idx_rv = dict([(value, key) for key, value in drug_idx.iteritems()])
        plates_idx_rv = dict([(value, key) for key, value in plates_idx.iteritems()])

        cells_no = len(cell_idx)
        drugs_no = len(drug_idx)
        plates_no = len(plates_idx)

        depth_limiter = 7

        storage = np.empty((cells_no, drugs_no, depth_limiter, 10, 3))
        storage.fill(np.NaN)

        background = np.empty((cells_no, drugs_no, depth_limiter, 4))
        background.fill(np.NaN)

        t0_median = np.empty((cells_no, drugs_no, depth_limiter))
        t0_median.fill(np.NaN)

        t0_background = np.empty((cells_no, drugs_no, depth_limiter))
        t0_background.fill(np.NaN)

        tf_background = np.empty((cells_no, drugs_no, depth_limiter))
        tf_background.fill(np.NaN)

        background_noise = np.empty((plates_no, 2))
        background_noise.fill(np.NaN)

        cl_drug_replicates = np.zeros((cells_no, drugs_no))

        with open(path.join(pth, fle)) as src:
            rdr = reader(src, dialect='excel-tab')
            test_array = rdr.next()
            supporting_functions.broadcast(test_array[6:36])
            for row in rdr:
                cell_no = cell_idx[row[0]]
                drug_no = drug_idx[(row[1], float(row[47]))]
                plate_no = plates_idx[row[2]]
                depth_index = min(cl_drug_replicates[cell_no, drug_no], depth_limiter-1)
                storage[cell_no, drug_no, depth_index, :, :] = supporting_functions.broadcast(row[6:36])
                background[cell_no, drug_no, depth_index, :] = supporting_functions.lgi(row, [4, 5, 36, 37])
                t0_median[cell_no, drug_no, depth_index] = row[38]
                t0_background[cell_no, drug_no, depth_index] = np.mean(
                    supporting_functions.lgi(row, [4, 5]).astype(np.float64)).tolist()
                tf_background[cell_no, drug_no, depth_index] = np.mean(
                    supporting_functions.lgi(row, [36, 37]).astype(np.float64)).tolist()
                background_noise[plate_no, :] = np.abs(
                    supporting_functions.lgi(row, [4, 36]).astype(np.float64) -
                    supporting_functions.lgi(row, [5, 37]).astype(
                        np.float64))
                cl_drug_replicates[cell_no, drug_no] += 1

        cl_drug_replicates[cl_drug_replicates < 1] = np.nan

        alpha_bound = np.percentile(rm_nans(background_noise), 100 - alpha_bound_percentile)
        std_of_tools = np.percentile(rm_nans(background_noise), 66)

        background = supporting_functions.p_stabilize(background, 0.5)
        t0_background = supporting_functions.p_stabilize(t0_background, 0.5)
        tf_background = supporting_functions.p_stabilize(tf_background, 0.5)

        storage_dblanc = storage - tf_background[:, :, :, np.newaxis, np.newaxis]

        self.header_line = header
        self.cell_line_2_idx = cell_idx
        self.drug_2_idx = drug_idx
        self.idx_2_cell_line = cell_idx_rv
        self.idx_2_drug = drug_idx_rv
        self.raw_data = storage_dblanc           # cell_line, drug, concentration -> 3 replicates
        self.background = background             # cell_line, drug -> (T0_1, T0_2, T_final, T_final) backgrounds
        self.t0_background = t0_background
        self.t_f_background = tf_background
        self.t0_median = t0_median              # for each cell_line and drug contains T0
        self.alpha_bound = alpha_bound          # lower significance bound
        self.std_of_tools = std_of_tools
        self.background_noise = background_noise
        self.drug_versions = dict(drug_versions)      # drug names + concentrations versions
        self.cl_drug_replicates = cl_drug_replicates

    def return_relevant_values(self):
        render_dict = {}
        for _property, value in vars(self).iteritems():
            render_dict[str(_property)] = value
        return render_dict

    def retrieve(self, cell, drug, correct_background=True, correct_lower_boundary=True):
        drug_c_array = np.array([0]+[2**_i for _i in range(0, 9)])*0.5**8

        def helper_round(t_container):
            t_container_vals = [np.repeat(t_container[cell_n, _drug_n][:, np.newaxis], 10, axis=1)
                                for _drug_n in drugs_nos]
            t_container_vals = supporting_functions.block_fusion(t_container_vals)
            t_container_vals = t_container_vals[:, c_argsort]
            return t_container_vals

        cell_n = self.cell_line_2_idx[cell]
        retained_drugs = [drug_v for drug_v in self.drug_versions[drug]]

        drugs_nos = [self.drug_2_idx[drug_v] for drug_v in retained_drugs]
        drug_vals = [self.raw_data[cell_n, drug_n].copy() for drug_n in drugs_nos]

        drug_c = [drug_v[1]*drug_c_array for drug_v in retained_drugs]
        anchor = np.min(np.array([drug_v[1] for drug_v in retained_drugs]))
        anchor /= 2.**10
        drug_vals = supporting_functions.block_fusion(drug_vals)

        drug_c = np.hstack(drug_c)
        c_argsort = np.argsort(drug_c)

        drug_c = drug_c[c_argsort]
        drug_vals = drug_vals[:, c_argsort, :]

        t0_bck_vals = helper_round(self.t0_background.copy())
        tf_bck_vals = helper_round(self.t_f_background.copy())
        t0_vals = helper_round(self.t0_median.copy())
        noise_dispersion = [helper_round(self.background[:, :, :, i].copy()) for i in range(0, 4)]
        noise_dispersion = supporting_functions.estimate_differences(noise_dispersion)

        if correct_background:
            drug_vals -= tf_bck_vals[:, :, np.newaxis]
            t0_vals -= t0_bck_vals[:, :]

        if correct_lower_boundary:
            drug_vals = supporting_functions.get_boundary_correction(drug_vals, self.std_of_tools)

        # print 'drug_vals', drug_vals
        # print 'drug_c', drug_c
        # print 't0_vals', t0_vals

        return drug_vals, drug_c, t0_vals, noise_dispersion, anchor

    def run_quality_control(self):
        pass


class Gi50Reader(object):

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

        cell_idx = supporting_functions.index(cells)
        drug_idx = supporting_functions.index(drugs)

        cell_idx_rv = dict([(value, key) for key, value in cell_idx.iteritems()])
        drug_idx_rv = dict([(value, key) for key, value in drug_idx.iteritems()])

        self.cell_line_2_idx = cell_idx
        self.drug_2_idx = drug_idx
        self.idx_2_cell_line = cell_idx_rv
        self.idx_2_drug = drug_idx_rv
        self.gi_50 = np.array(data_matrix)

    def retrieve(self, cell_line, drug):
        if cell_line in self.cell_line_2_idx and drug in self.drug_2_idx:
            return self.gi_50[self.cell_line_2_idx[cell_line], self.drug_2_idx[drug]]
        else:
            return np.NaN

    def run_qc(self):
        pass


class ClassificationReader(object):

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

        cell_idx = supporting_functions.index(cells)

        cell_idx_rv = dict([(value, key) for key, value in cell_idx.iteritems()])

        self.cassificant_index = cell_idx
        self.header = header
        self.classificant_index_rv = cell_idx_rv
        self.markers = markers


def fragmented_round(cell_line, drug, color='black', plot_type=10, injected_anchor=None):

    render_type = int(str(plot_type)[0])
    normalization_type = int(str(plot_type)[1])

    plot_touched = False

    tf_od, concentrations, t0_median, noise_dispersion, anchor = hr.retrieve(cell_line, drug)

    if injected_anchor is not None:
        anchor = injected_anchor

    clean_mask = supporting_functions.clean_nans(tf_od)

    if not np.any(clean_mask):
        return plot_touched, (None, None, None), (None, None, None, None)

    tf_od, t0_median = (tf_od[clean_mask, :, :], t0_median[clean_mask, :])
    tf_corrected, means_arr, errs_arr, unique_concs_stack =\
        supporting_functions.correct_plates(tf_od, concentrations, hr.std_of_tools)

    if render_type == 1:
        plot_touched = True
        plot_drawings.raw_plot(tf_corrected, tf_od, concentrations, hr.std_of_tools, color=color)
        plot_drawings.vector_summary_plot(means_arr, errs_arr, unique_concs_stack, anchor, color=color)
        # plt.show()

    clean_mask = supporting_functions.clean_nans(tf_corrected)

    if not np.any(clean_mask):
        return plot_touched, (None, None, None), (None, None, None, None)

    tf_corrected, t0_corrected = (tf_corrected[clean_mask, :, :], t0_median[clean_mask, :])
    means_arr, errs_arr = (means_arr[clean_mask, :], errs_arr[clean_mask, :])
    saved_means_arr, saved_errs_arr = means_arr.copy(), errs_arr.copy()

    norm_factor = None

    if normalization_type == 0:
        norm_factor = supporting_functions.retrieve_normalization_factor(means_arr)

    if normalization_type == 1:
        norm_factor = supporting_functions.retrieve_normalization_factor(t0_corrected)

    norm_plate, norm_means, norm_errs, norm_std_of_tools =\
        supporting_functions.normalize(tf_corrected, means_arr, errs_arr, hr.std_of_tools, norm_factor)

    if render_type == 2:
        plot_touched = True
        plot_drawings.vector_summary_plot(norm_means, norm_errs, unique_concs_stack, anchor, color=color)
        # plt.show()

    means, errs, unique_concs = supporting_functions.combine(norm_plate, concentrations, np.max(norm_std_of_tools))

    if render_type == 3:
        plot_touched = True
        plot_drawings.summary_plot(means, errs, unique_concs, anchor, color=color)
        # plt.show()

    norm_factor = supporting_functions.retrieve_normalization_factor(t0_corrected)
    norm_plate, norm_means, norm_errs, norm_std_of_tools =\
        supporting_functions.normalize(tf_corrected, saved_means_arr, saved_errs_arr, hr.std_of_tools, norm_factor)

    return plot_touched, (means, errs, unique_concs), (norm_means, norm_errs)


def compare_to_htert(cell_line, drug, standardized, plot_type=1):
    plt.title('%s, %s' % (cell_line, drug))

    print '{0:15}'.format(cell_line),
    print '{0:20}'.format(drug),
    print '\t',

    _, _, _, _, r_anchor = hr.retrieve('184A1', drug)
    _, _, _, _, g_anchor = hr.retrieve('184B5', drug)
    _, _, _, _, b_anchor = hr.retrieve(cell_line, drug)

    anchor = np.nanmin(np.array([r_anchor, g_anchor, b_anchor]))

    rt, _, _ = fragmented_round('184A1', drug, 'red', plot_type, anchor)
    print 'rt',
    gt, _, _ = fragmented_round('184B5', drug, 'green', plot_type, anchor)
    print 'gt',
    bt, _, _ = fragmented_round(cell_line, drug, 'black', plot_type, anchor)
    print 'bt',

    if not bt:
        print '-'
        return ''

    plt.gcf().set_size_inches(25, 15, forward=True)
    plt.autoscale(tight=True)
    plt.legend()

    cell_line = slugify(unicode(cell_line))
    drug = slugify(unicode(drug))

    supporting_functions.safe_dir_create('../analysis_runs/%s/vrac/' % supporting_functions.type_map(plot_type))
    plt.savefig('../analysis_runs/%s/vrac/%s - %s.png' % (supporting_functions.type_map(plot_type), cell_line, drug))

    supporting_functions.safe_dir_create('../analysis_runs/%s/by_drug/%s/' % (supporting_functions.type_map(plot_type), drug))
    plt.savefig('../analysis_runs/%s/by_drug/%s/%s.png' % (supporting_functions.type_map(plot_type), drug, cell_line))

    supporting_functions.safe_dir_create('../analysis_runs/%s/by_cell_line/%s/' % (supporting_functions.type_map(plot_type), cell_line))
    plt.savefig('../analysis_runs/%s/by_cell_line/%s/%s.png' % (supporting_functions.type_map(plot_type), cell_line, drug))

    print '+'
    return ''


def graphics_loop(plot_type=10):

    def nan(_drug_n):
            return np.all(np.isnan(hr.raw_data[cell_n, _drug_n]))

    print 'starting graphics loop with the following parameters: %s' % (plot_type)

    for drug in hr.drug_versions.keys():
        for cell_line, cell_n in hr.cell_line_2_idx.iteritems():
            if not cell_line in ['184A1', '184B5']:
                if [drug_v for drug_v in hr.drug_versions[drug] if not nan(hr.drug_2_idx[drug_v])]:
                    compare_to_htert(cell_line, drug, True, plot_type)
                    plt.clf()


def computational_loop(plot_type=40):

    def nan(_drug_n):
            return np.all(np.isnan(hr.raw_data[cell_n, _drug_n]))

    print 'starting computational loop'

    memory_dict = {}
    drug2cell_line = defaultdict(list)
    cell_line2drug = defaultdict(list)

    for drug in hr.drug_versions.keys():
        for cell_line, cell_n in hr.cell_line_2_idx.iteritems():
            if [drug_v for drug_v in hr.drug_versions[drug] if not nan(hr.drug_2_idx[drug_v])]:
                print cell_line, drug
                _, collapsed, stacked = fragmented_round(cell_line, drug, plot_type=plot_type)
                if not collapsed[0] == None:
                    memory_dict[drug, cell_line] = (collapsed, stacked)
                    drug2cell_line[drug].append(cell_line)
                    cell_line2drug[cell_line].append(drug)

    dump(memory_dict, open('../analysis_runs/memdict.dmp', 'w'))
    dump(drug2cell_line, open('../analysis_runs/drug2cell_line.dmp', 'w'))
    dump(cell_line2drug, open('../analysis_runs/cell_line2drug.dmp', 'w'))


def test_GI_50_reader():
    tr = Gi50Reader('C:\\Users\\Andrei\\Desktop', 'sd02.tsv')
    print tr.retrieve('HCC1954', 'XRP44X')


if __name__ == "__main__":
    hr = RawDataReader('C:\\Users\\Andrei\\Desktop', 'gb-breast_cancer.tsv')
    tr = Gi50Reader('C:\\Users\\Andrei\\Desktop', 'sd05-bis.tsv')

    # print fragmented_round('600MPE', 'GSK1838705', plot_type=30)
    # plt.show()

    # graphics_loop(30)
    computational_loop(40)
    # 1x-3x = various debug renders; 4x - no debug render;
    # x1 - no normalization; x0 - normalize to starting
