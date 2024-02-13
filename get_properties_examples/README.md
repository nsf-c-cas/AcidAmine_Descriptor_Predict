This folder contains the scripts used to collect conformer properties and molecular descriptors for libraries of carboxylic acids, primary alkyl amines, and secondary alkyl amines.

The [Get Properties repository](https://github.com/SigmanGroup/Get_Properties/tree/main) details the automated and adaptable moiety-specific descriptor collection from a SMARTS string input. Instructions for setting up the conda environment and how to run the code are provided. This code was adapted for the substrates and descriptors of interest in this study.

The folder is organized as follows:

`get_acid_properties_example/` contains scripts for obtaining descriptors included in the library from carboxylic acid Gaussian 16 calculations. Includes an additional function to extract buried volume divided into hemispheres.
`get_primary_amine_properties_example/` contains scripts for obtaining descriptors included in the library from primary amine Gaussian 16 calculations. Includes an additional function to extract lone pair energy and occupancy.
`get_secondary_amine_properties_example` contains scripts for obtaining descriptors included in the library from secondary amine Gaussian 16 calculations. Includes an additional function to extract lone pair energy and occupancy.