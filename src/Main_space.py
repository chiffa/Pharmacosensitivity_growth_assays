from readers import RawDataReader
from plot_drawings import pretty_gradual_plot
import matplotlib as mlb

mlb.rcParams['figure.figsize'] = (19, 10)
current_dataset = RawDataReader('C:\\Users\\Andrei\\Desktop', 'gb-breast_cancer.tsv')
drug = 'Paclitaxel'
example = current_dataset.raw_data[:, current_dataset.drug_2_idx[drug], :, :]
example_concs = current_dataset.concentrations[:, current_dataset.drug_2_idx[drug], :]
pretty_gradual_plot(example, example_concs, current_dataset.idx_2_cell_line, drug,
                    blank_line=current_dataset.alpha_bound)
