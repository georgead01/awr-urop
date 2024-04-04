import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import ml4chem
import mdtraj
from data_proc import get_molecules, get_trajectory
from ml4chem.data.handler import Data
from ase.io import Trajectory
import ase.io

from ml4chem.atomistic.models.autoencoders import VAE
from ase.calculators.abinit import Abinit, AbinitProfile
# from ase.calculators.abacus import Abacus, AbacusProfile
abinit = '/usr/local/bin/abinit'
profile = AbinitProfile(argv=['mpirun','-n','2',abinit])



# get data

dft_path = 'DFT_results'
molecules = os.listdir(dft_path)
train_data = []

for mol in molecules:
    if mol != 'extract_data.sh':
        traj_path = os.path.join(dft_path, f'{mol}/{mol}_DFT.traj')
        traj_read = ase.io.Trajectory(traj_path)
        atoms = traj_read[:]
        calc = Abinit(profile=profile, ntype=1, ecutwfc=50, scf_nmax=50, smearing_method='gaussian', smearing_sigma=0.01, basis_type='pw', ks_solver='cg', calculation='scf', pp=pp, basis=basis, kpts=kpts)
        atoms.calc = calc
        train_data.append(*atoms)

data_handler = Data(train_data, purpose='training')

hiddenlayers = {"encoder": (20, 10, 4), "decoder": (4, 10, 20)}
activation = "tanh"
vae = VAE(hiddenlayers=hiddenlayers, activation=activation, variant="multivariate")
data_handler.get_unique_element_symbols(train_data, purpose='training')
vae.prepare_model(120, 120, data=data_handler)

vae.train()
print(vae)