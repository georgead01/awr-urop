import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import ml4chem
import mdtraj
from data_proc import get_molecules, get_trajectory
from ml4chem.data.handler import Data

# get data

dft_path = 'DFT_results'
molecules = os.listdir(dft_path)
train_data = []

for mol in molecules:
    if mol != 'extract_data.sh':
        train_data.append(get_trajectory(mol))

data_handler = Data(train_data, purpose='training')