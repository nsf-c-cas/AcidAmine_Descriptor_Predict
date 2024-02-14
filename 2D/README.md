This folder contains the necessary code to:

1. Apply pre-trained GINE models to predict acid and amine descriptors from SMILES strings.
2. Train or retrain GINE models on the acid or amine desciptors reported in our manuscript.
The folder is organized as follows:

`datasets/` contains Pytorch and Pytorch-Geometric dataset / dataloader / sampler classes. You should not have to modify these files.
`models/` contains the implementation of GINE, a state-of-the-art graph neural network that strikes a good balance between speed and accuracy. You also should not have to modify these files.
`predictions.ipynb` is a notebook that demonstrates how to predict acid/amine descriptors from SMILES strings.

Additional material on `trained_models`, `data` for training, `test predictions` can be obtianed in the following [Figshare](https://doi.org/10.6084/m9.figshare.25213742.v3).

## Setting up conda environment

`environment.yml` contains a conda environment with all the necessary dependencies. If you're lucky, the simplest way to set up the environment is to call:

`conda env create -f environment.yml`

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

## Steps to run example prediction

To utilise the developed using `example_prediciton.ipynb` please preform the following steps

1. Download the trained models (2D/trained_models) from [Figshare](https://doi.org/10.6084/m9.figshare.25213742.v3)
2. Stored the downloaded folder in the same folder as `example_prediciton.ipynb`.
3. Excecute the the necessary cells in the jupyter notebook for prediction of acid/amine properties.

## Steps to re-train the model

To run re-training please perform the following steps

1. Download the dataset for training models (2D/data) from [Figshare](https://doi.org/10.6084/m9.figshare.25213742.v3)
1. Download the trained models (2D/trained_models) from [Figshare](https://doi.org/10.6084/m9.figshare.25213742.v3) if you would like to start from pre-trained models
3. Each surrogate descriptor prediction model can be (re)trained by calling `train.py` like so:

`python train.py OUT_DIR JOB_NAME DATAFRAME_PATH PROP_TYPE KEEP_HYDROGENS MODEL_TYPE PATH_TO_PRETRAINED_MODEL BATCH_SIZE LEARNING_RATE SEED LOSS_TYPE`

where:

    - OUT_DIR (str) specifies the ouput directory for the log file, saved model checkpoints, training and validation losses, etc.
    
    - JOB_NAME (str) specifies the name of property for training. The output from training for this propertry will be stored in `OUT_DIR/JOB_NAME`.
    
    - DATAFRAME_PATH (str) specifies the path to the downloaded data from figshare. For example `data/HOMO_Boltz.csv`
    
    - PROP_TYPE ('mol', 'atom', or 'bond') specifies the type of descriptor.
    
    - KEEP_HYDROGENS (int) specifies whether to keep explicit hydrogens in the conformers during training.
        
        - KEEP_HYDROGENS = 0 indicates to remove all explicit hydrogens. This cannot be done for atom-level descriptors that are specific to a hydrogen (e.g., hydrogen NBO charge).
        
        - KEEP_HYDROGENS = 1 indicates to keep all explicit hydrogens in the conformers.
        
        - KEEP_HYDROGENS = 2 indicates to only keep the hydrogens within the acid or amine functional groups.
        Because the hydrogen geometries optimized by MMFF94 are poor proxies for DFT-level geometries, we recommend using KEEP_HYDROGENS = 2 for hydrogen-specific atom-level descriptors, and KEEP_HYDROGENS = 0 for all other descriptors. Using KEEP_HYDROGENS = 1 slows down training and does not usually result in better performance.
    
    - MODEL_TYPE (str) the current implementation for 2D models allows only for GINE architecture. 
        
    - PATH_TO_PRETRAINED_MODEL (str) specifies a path to pretrained model weights, useful for fine-tuning or transfer-learning purposes. PATH_TO_PRETRAINED_MODEL can also be set to 'checkpoint' in order to resume a previous training run. Otherwise, if you're training from scratch, just set PATH_TO_PRETRAINED_MODEL = 'none'.
    
    - BATCH_SIZE (int) sets the batch size during training. We used 128.
    
    - LEARNING_RATE (float) sets the constant learning rate during training. We used 0.0001.
    
    - SEED (int) defines a random seed for training.
    
    - LOSS_TPYE (str) species the loss type used in training. Current implementation will all for usage of MSE or MAE.
    




