import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import ml4chem
import mdtraj
from data_proc import get_molecules, get_trajectory
from ml4chem.data.handler import Data
from ase.io import Trajectory
import ase.io

from ml4chem.atomistic.models.autoencoders import VAE
from ase.calculators.abc import ABC
from ase.calculators.abinit import Abinit

# get data

dft_path = 'DFT_results'
molecules = os.listdir(dft_path)
train_data = []

for mol in molecules:
    if mol != 'extract_data.sh':
        traj_path = os.path.join(dft_path, f'{mol}/{mol}_DFT.traj')
        traj_read = ase.io.Trajectory(traj_path)
        atoms = traj_read[:]
        for atom in atoms:
            calc = Abinit(label=str(atom),
            pseudo_dir='pbe_s_sr',
            ecut=10.0,     # Plane-wave energy cutoff in Hartree
            kpts=(4, 4, 4),  # k-points mesh
            toldfe=1.0e-6,  # SCF convergence threshold
            nshiftk=4,      # Number of k-point shifts
            shiftk=[(0.5, 0.5, 0.5), (0.5, 0.0, 0.0),
                    (0.0, 0.5, 0.0), (0.0, 0.0, 0.5)])
            atom.calc = calc
            train_data.append(atom)

data_handler = Data(train_data, purpose='training')

hiddenlayers = {"encoder": (20, 10, 4), "decoder": (4, 10, 20)}
activation = "tanh"
vae = VAE(hiddenlayers=hiddenlayers, activation=activation, variant="multivariate")
data_handler.get_unique_element_symbols(train_data, purpose='training')
vae.prepare_model(120, 120, data=data_handler)

vae.train()
print(vae)