__author__ = 'Andrei'

from readers import historical_reader
from supporting_functions import pretty_gradual_plot
import matplotlib as mlb

mlb.rcParams['figure.figsize'] = (19, 10)
hr = historical_reader('C:\\Users\\Andrei\\Desktop', 'gb-breast_cancer.tsv')
drug = 'Paclitaxel'
example = hr.storage[:, hr.drug_idx[drug], :, :]
example_concs = hr.concentrations[:, hr.drug_idx[drug], :]
pretty_gradual_plot(example, example_concs, hr.cell_idx_rv, drug, blank_line=hr.sensible_max)