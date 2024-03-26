import os
import mdtraj

dft_path = 'DFT_results'
files = os.listdir(dft_path)

def get_molecule(mol):
    if not mol in files:
        raise KeyError(f'no molecule {mol} in folder')
    
    file_path = os.path.join(dft_path, f'{mol}_DFT_trj.xyz')
    return file_path

def get_molecules():
    mols = []
    for mol in files:
        if mol != 'extract_data.sh':
            mols.append(mol)

    return mols

def get_trajectory(mol):
    if not mol in files:
        raise KeyError(f'no molecule {mol} in folder')
    
    file_path = os.path.join(dft_path, f'{mol}/{mol}_DFT_topology.pdb')
    return mdtraj.load(file_path)