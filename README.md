# awr-urop
 code for AWR UROP


## script.py

where the main script lives. issues fixed so far:

- trun files into trajectory files.
- added calculators to molecules.

issues to be fixed:

- pseudopotentials.

## convert_to_traj.py

reads xyz files from the DFT_results folder and converts to trajectory files.

## data_proc.py

contains some helper functions to read data- might need some updating to read trajectory files.

## VAE.py

contains a VAE class - doesn't work as of now.

## SMILES_Big_Data_Set.csv

a dataset of SMILES code for training purposes (might be helpful especially if we use a SMILES VAE).

## DFT_results

DFT files + trajectory and pdb conversions.

## pbe_s_sr

pseudopotential directory - might need to install a different one.