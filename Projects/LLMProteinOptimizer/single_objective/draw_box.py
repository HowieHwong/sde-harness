

import yaml

with open('/home/jhe/Molleo_Protein/single_objective/main/molleo/results/results_EA_AAV_1.yaml', 'r') as f:
    loaded_data_EA = yaml.load(f, Loader=yaml.Loader) 
with open('/home/jhe/Molleo_Protein/single_objective/main/molleo/results/results_Llama3_AAV_1.yaml', 'r') as f:
    loaded_data_Llama3 = yaml.load(f, Loader=yaml.Loader) 
print(len(loaded_data_EA))
print(len(loaded_data_Llama3))