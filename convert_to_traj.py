import os
import mdtraj
import ase.io

dft_path = 'DFT_results'
files = os.listdir(dft_path)

for file in files:
    if file != 'extract_data.sh':
        print(f'converting {file} xyz to traj...')
        xyz_file = os.path.join(dft_path, f'{file}/{file}_DFT.xyz')
        traj_path = os.path.join(dft_path, f'{file}/{file}_DFT.traj')

        atoms = ase.io.read(xyz_file)
        writer = ase.io.trajectory.TrajectoryWriter(traj_path)
        writer.write(atoms)
        writer.close()

        print(f'{file} conversion complete!')