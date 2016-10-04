from readers import raw_data_reader
from plot_drawings import pretty_gradual_plot
import matplotlib as mlb

mlb.rcParams['figure.figsize'] = (19, 10)
current_dataset = raw_data_reader('C:\\Users\\Andrei\\Desktop', 'gb-breast_cancer.tsv')
drug = 'Paclitaxel'
example = current_dataset.storage[:, current_dataset.drug_idx[drug], :, :]
example_concs = current_dataset.concentrations[:, current_dataset.drug_idx[drug], :]
pretty_gradual_plot(example, example_concs, current_dataset.cell_idx_rv, drug, blank_line=current_dataset.alpha_bound)