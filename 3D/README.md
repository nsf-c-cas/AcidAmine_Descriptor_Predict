## Overview

This repository contains the necessary code to:
1. Apply pre-trained DimeNet++ models to predict acid and amine descriptors from SMILES strings.
2. Train or retrain DimeNet++ models on the acid or amine desciptors reported in our manuscript.
2. Test or retest DimeNet++ models on the acid or amine desciptors reported in our manuscript.

Before training or evaluating the models reported in the paper, the following directories need to be downloaded from the [Figshare](https://doi.org/10.6084/m9.figshare.25213742.v1) and added to this directory under `data/`, `trained_models/`, and `external_validation_data/`:

- `3D/data/`

- `3D/trained_models/`

- `3D/external_validation_data/`


This repository is organized as follows:

`3D/data/` contains the MMFF94-level conformer ensembles (from RDKit) encoded by the DimeNet++ models. The subdirectories `data/acid`, `data/primary_amine`, `data/secondary_amine`, and `data/combined_amine` include training-ready dataframes (stored as .csv files) that contain the regression targets for each descriptor that was predicted by DimeNet++ in our manuscript.

`3D/trained_models/` contains pre-trained Pytorch model weights for regression models trained on each acid or amine descriptor.

`3D/external_validation_data/` contains the data (SMILES and DFT-computed descriptors) for the external validation sets.

`datasets/` contains Pytorch and Pytorch-Geometric dataset class that handles all the graph batching, atom featurization, etc. needed for training or using the DimeNet++ models.

`models/` contains the implementation of DimeNet++ (adapted from that provided by Pytorch Geometric), a state-of-the-art 3D graph neural network that strikes a good balance between speed and accuracy.

`utils.py` contains supporting functions for conformer generation, model inference, substructure identification, etc.

`train.py` contains a training script that can be used to (re)train any of the DimeNet++ models reported in our manuscript, from the command line.

`train.py` contains a testing script that can be used to (re)-evaluate any of the DimeNet++ models reported in our manuscript, from the command-line.

`test_acids.ipynb` contains a notebook to evaluate the DimeNet++ models on the acids test set for each descriptor.
`test_amines.ipynb` contains a notebook to evaluate the DimeNet++ models on the primary amines test set for each descriptor.
`test_sec_amines.ipynb` contains a notebook to evaluate the DimeNet++ models on the secondary amines test set for each descriptor.

`predictions_external_validation.ipynb` is a user-friendly notebook that demonstrates how to predict acid/amine descriptors from SMILES strings, using the DimeNet++ models. This notebook was applied to predict the descriptors of the external validation set in our manuscript.


## Setting up conda environment

`environment_cpu.yml` contains a conda environment with all the necessary dependencies. With luck, the simplest way to set up the environment is to call:

`conda env create -f environment_cpu.yml`

from the command line. Oftentimes, this doesn't work when transferring the environment across different systems. Instead, you can create the environment manually by installing the following key packages:

- python=3.10.8

- notebook=6.5.2

- numpy=1.23.3

- pandas=1.4.4

- rdkit=2022.03.2

- tqdm==4.64.1

- scipy==1.9.3

- scikit-learn==1.1.3 (shouldn't be necessary, but is oftentimes installed alongside numpy/scipy automatically)

- pytorch=1.13.0

- torchaudio=0.13.0

- torchvision=0.14.0

- torch-cluster==1.6.0

- torch-sparse==0.6.15

- torch-spline-conv==1.2.1

- torch-scatter==2.0.9

- torch-geometric==2.1.0

Installing `pytorch` and `torch-geometric` may cause some headaches. It may be helpful to follow instructions here:

https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
https://pytorch.org/get-started/locally/

The versions can *really* matter when installing `pytorch` and `torch-geometric`, so try to install the exact versions listed above.


Activate the environment by:

`conda activate amide_coupling`
or 
`source activate amide_coupling`


## Instructions for (re)-training

Each surrogate descriptor prediction model can be (re)trained by calling `train.py` like so:

`python train.py OUT_DIR PATH_TO_CONFORMERS_CSV_FILE PATH_TO_DESCRIPTOR_REGRESSION_TARGETS PROP_TYPE N_CONFORMERS KEEP_HYDROGENS PATH_TO_PRETRAINED_MODEL BATCH_SIZE LEARNING_RATE SEED STARTING_EPOCH USE_ATOM_FEATURES`


where:

    - OUT_DIR (str) specifies the ouput directory for the log file, saved model checkpoints, training and validation losses, etc.
    
    - PATH_TO_CONFORMERS_CSV_FILE (str) specifies the csv file that contains the input conformers encoded by the model during training or inference. See `data/3D_model_acid_rdkit_conformers.csv` for an example on how this file should be constructed.
    
    - PATH_TO_DESCRIPTOR_REGRESSION_TARGETS (str) specifies the csv file that contains the regression targets (e.g., DFT-level descriptors). This file will get merged with the conformer_csv prior to training, wherein each conformer in the conformer_csv will be mapped to the same descriptor value. For atom-level descriptors, one field in this csv must be "atom_index", which specifies the index of the query atom associated with the descriptor. For bond-level descriptors, one field in this csv must be "bond_atom_tuple" with specifies two atom indices of the bond associated with the descriptor. These atom indices must be defined with respect to the conformers in conformer_csv. See `acid/atom/C1_Vbur_Boltz.csv` and `acid/bond/IR_freq_Boltz.csv` for examples of how these files should be constructed.
    
    - PROP_TYPE ('mol', 'atom', or 'bond') specifies the type of descriptor.
    
    - N_CONFORMERS (int) sets the number of input conformers to be down-sampled or up-sampled for each molecule, ensuring a balanced dataset. If molecule A has 5 conformers and molecule B has 18 conformers, setting N_CONFORMERS to 10 will sample each conformer from molecule A twice, but only 10/18 conformers from molecule B. Conformers are sampled prior to training and are fixed throughout training. We used N_CONFORMERS=10 for acids and N_CONFORMERS=20 for amines.
    
    - KEEP_HYDROGENS (int) specifies whether to keep explicit hydrogens in the conformers during training.
        
        - KEEP_HYDROGENS = 0 indicates to remove all explicit hydrogens. This cannot be done for atom-level descriptors that are specific to a hydrogen (e.g., hydrogen NBO charge).
        
        - KEEP_HYDROGENS = 1 indicates to keep all explicit hydrogens in the conformers.
        
        - KEEP_HYDROGENS = 2 indicates to only keep the hydrogens within the acid or amine functional groups.
        Because the hydrogen geometries optimized by MMFF94 are poor proxies for DFT-level geometries, we recommend using KEEP_HYDROGENS = 2 for hydrogen-specific atom-level descriptors, and KEEP_HYDROGENS = 0 for all other descriptors. Using KEEP_HYDROGENS = 1 slows down training and does not usually result in better performance.
    
    
    - PATH_TO_PRETRAINED_MODEL (str) specifies a path to pretrained model weights, useful for fine-tuning or transfer-learning purposes. PATH_TO_PRETRAINED_MODEL can also be set to 'checkpoint' in order to resume a previous training run. Otherwise, if you're training from scratch, just set PATH_TO_PRETRAINED_MODEL = 'none'.
    
    - BATCH_SIZE (int) sets the batch size during training. We used 128.
    
    - LEARNING_RATE (float) sets the constant learning rate during training. We used 0.0001.
    
    - SEED (int) defines a random seed for training.
    
    - STARTING_EPOCH (int) species the starting epoch number for training. Most of the time, this will be 0. If you're resuming a training run, you might consider setting this to the next epoch from where you left off.
    
    - USE_ATOM_FEATURES (int) indicates whether to use RDKit atom features for node featurization (USE_ATOM_FEATURES=1), or whether to use DimeNet++'s default atom featurization (USE_ATOM_FEATURES=0). We used USE_ATOM_FEATURES = 1.
    

Examples:

For training DimeNet++ to predict the Boltzmann-averaged NBO charge for the C1 atom of the acids:
        
`python train.py jobs_example_acid_C1_NBO_charge_Boltz/ data/3D_model_acid_rdkit_conformers.csv data/acid/atom/C1_NBO_charge_Boltz.csv atom 10 0 none 128 0.0001 0 0 1`

For training DimeNet++ to predict the IR frequency of the lowest-energy conformer of the acids: 

`python train.py jobs_example_acid_IR_freq_lowE/ data/3D_model_acid_rdkit_conformers.csv data/acid/bond/IR_freq_lowE.csv bond 10 0 none 128 0.0001 0 0 1`

For training DimeNet++ to predict the maximum NBO charge on the hydrogen for the secondary amines, using 20 input conformers for each molecule:

`python train.py jobs_example_secamine_NBO_charge_H_max/ data/3D_model_secondaryamine_rdkit_conformers.csv data/secondary_amine/atom/NBO_charge_H_max.csv atom 20 2 none 128 0.0001 0 0 1`

For training DimeNet++ to predict the minimum dipole moment of the combined (primary + secondary) amines:

`python train.py jobs_example_combinedamine_dipole_min/ data/3D_model_combinedamine_rdkit_conformers.csv data/combined_amine/mol/dipole_min.csv mol 20 0 none 128 0.0001 0 0 1`

For training DimeNet++ to predict the Sterimol-B5 value of the lowest energy conformer of the primary amines:

`python train.py jobs_example_primaryamine_Sterimol_B5_lowE/ data/3D_model_primaryamine_rdkit_conformers.csv data/primary_amine/bond/Sterimol_B5_lowE.csv bond 20 0 none 128 0.0001 0 0 1`


## Instructions for (re)-testing

Each surrogate descriptor prediction model can be (re)tested on our test sets by calling `test.py` like so:

`python test.py PATH_TO_CONFORMERS_CSV_FILE PATH_TO_DESCRIPTOR_REGRESSION_TARGETS PROP_TYPE KEEP_HYDROGENS PATH_TO_PRETRAINED_MODEL USE_ATOM_FEATURES`

where the command-line arguments are defined in the same way as for `train.py`.

Examples:

`python test.py data/3D_model_acid_rdkit_conformers.csv data/acid/atom/C1_NBO_charge_Boltz.csv atom 0 trained_models/acids/C1_NBO_charge/boltz/model_best.pt 1`

`python test.py data/3D_model_acid_rdkit_conformers.csv data/acid/atom/C1_Vbur_Boltz.csv atom 0 trained_models/acids/C1_Vbur/boltz/model_best.pt 1`

`python test.py data/3D_model_acid_rdkit_conformers.csv data/acid/atom/C1_Vbur_max.csv atom 0 trained_models/acids/C1_Vbur/max/model_best.pt 1`

`python test.py data/3D_model_acid_rdkit_conformers.csv data/acid/atom/C1_Vbur_min.csv atom 0 trained_models/acids/C1_Vbur/min/model_best.pt 1`

`python test.py data/3D_model_acid_rdkit_conformers.csv data/acid/atom/C1_Vbur_lowE.csv atom 0 trained_models/acids/C1_Vbur/min_E/model_best.pt 1`


## Notes:

- Training on a CPU can sometimes take ~2 days for training to converge *completely* (in terms of MAE on validation set), but often good performance is obtained after a few hours.

- Given the many models trained in the manuscript, minimal hyperparameter optimization was performed. It is likely that results on any individual descriptor could be slightly improved with more extensive hyperparameter search and optimization. Architectural hyperparameters may be changed by modifying our default call to DimeNetPlusPlus() in train.py.

