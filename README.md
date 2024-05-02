# awr-urop
 code for AWR UROP

## model:

our model consists of two components: a VAE and a FNN.

### VAE

the variational autoencoder's purpose is to encode the molecules in a latent space. once trained, we can use to encode/decode from and to the latent space.

### FNN

the fully connected neural network's purpose is to optimize the input for property, which then will be decoded. We will train the FNN to predict the desired properties by minimizing loss wrt to weights, and then find the optimal catalyst by optimizing the desired properties wrt the input.

## files:

### script.py

where the main script lives. issues fixed so far:

- trun files into trajectory files.
- added calculators to molecules.

issues to be fixed:

- pseudopotentials.

### convert_to_traj.py

reads xyz files from the DFT_results folder and converts to trajectory files.

### data_proc.py

contains some helper functions to read data- might need some updating to read trajectory files.

### VAE.py

contains a VAE class - doesn't work as of now.

### SMILES_Big_Data_Set.csv

a dataset of SMILES code for training purposes (might be helpful especially if we use a SMILES VAE).

### DFT_results

DFT files + trajectory and pdb conversions.

### pbe_s_sr

pseudopotential directory - might need to install a different one.

## resources:

- [ML4Chem](https://ml4chem.dev)
- [ASE (Atomic Simulation Environment)](https://wiki.fysik.dtu.dk/ase/index.html)
- [mol_opt](https://github.com/wenhao-gao/mol_opt)
- [MIT 6.S085](https://moldesign.github.io)
    - [Lab 4](https://colab.research.google.com/drive/1aB_Tn2q645GJVIirt7Zb_eoAbIth46HG)
    - [Lab 5](https://colab.research.google.com/drive/1yqMbgy05-JL68SIafZxk4bAXlQKkazAt)