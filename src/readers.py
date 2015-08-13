__author__ = 'Andrei'


from csv import reader
from os import path
from pprint import pprint
import numpy as np
from matplotlib import pyplot as plt
from supporting_functions import index, broadcast, make_comparator
from chiffatools.Linalg_routines import rm_nans

class historical_reader(object):

    def __init__(self, pth, fle):
        with open(path.join(pth, fle)) as src:
            cells = []
            drugs = []
            rdr = reader(src, dialect='excel-tab')
            header = rdr.next()
            for row in rdr:
                cells.append(row[0])
                drugs.append(row[1])

        cell_idx = index(set(cells))
        drug_idx = index(set(drugs))

        cell_idx_rv = dict([(value, key) for key, value in cell_idx.iteritems()])
        drug_idx_rv = dict([(value, key) for key, value in drug_idx.iteritems()])

        cellno = len(cell_idx)
        drugno = len(drug_idx)

        storage = np.empty((cellno, drugno, 10, 3))
        storage.fill(np.NaN)

        background = np.empty((cellno, drugno, 4))
        background.fill(np.NaN)

        concentrations = np.empty((cellno, drugno, 10))
        concentrations.fill(np.NaN)

        with open(path.join(pth, fle)) as src:
            rdr = reader(src, dialect='excel-tab')
            test_array = rdr.next()
            broadcast(test_array[6:36])
            for row in rdr:
                cell_no = cell_idx[row[0]]
                drug_no = drug_idx[row[1]]
                storage[cell_no, drug_no, :, :] = broadcast(row[6:36])
                background[cell_no, drug_no, :] = np.array([row[i] for i in [4, 5, 36, 37]])
                concentrations[cell_no, drug_no, :] = np.array([0]+row[39:48])

        median = np.percentile(rm_nans(background), 50)
        sensible_min = np.percentile(rm_nans(background), 2)
        sensible_max = np.percentile(rm_nans(background), 98)
        print sensible_min, median, sensible_max
        compare = make_comparator(max(sensible_max - median, median - sensible_min))


        self.header = header
        self.cell_idx = cell_idx
        self.drug_idx = drug_idx
        self.cell_idx_rv = cell_idx_rv
        self.drug_idx_rv = drug_idx_rv
        self.storage = storage
        self.background = background
        self.concentrations = concentrations
        self.median = median
        self.sensible_max = sensible_max
        self.sensible_min = sensible_min
        self.compare = compare


    def return_relevant_values(self):
        render_dict = {}
        for property, value in vars(self).iteritems():
            render_dict[str(property)] = value
        return render_dict


if __name__ == "__main__":
    pprint(historical_reader('C:\\Users\\Andrei\\Desktop', 'gb-breast_cancer.tsv').return_relevant_values())
