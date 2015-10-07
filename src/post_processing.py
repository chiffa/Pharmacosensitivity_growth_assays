__author__ = 'Andrei'

import numpy as np
from matplotlib import pyplot as plt
import plot_drawings as PD
from pickle import load

memdict = load(open('../analysis_runs/memdict.dmp', 'r'))
#[drug, cell_line] -> (means, mean errs, unique_concs), (mean_arr, err_arr, unique, T0)
drug2cell_line = load(open('../analysis_runs/drug2cell_line.dmp', 'r'))
cell_line2drug = load(open('../analysis_runs/cell_line2drug.dmp', 'r'))


if __name__ == '__main__':
    for drug, cell_lines in drug2cell_line.items():

        print cell_lines

        for cell_line in cell_lines:

            wrap, _ = memdict[drug, cell_line]
            means, errs, unique_c = wrap

            PD.summary_plot(means, errs, unique_c, anchor=1e-10, legend=cell_line)

        plt.show()
