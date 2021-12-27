# TCR-pMHC
### Prediction of TCR-pMHC Interactions Using Molecular Modeling and Recurrent Networks

This repository contains project data and code for prediction of binding between T-cell receptors (TCR) and peptides presented on major histocompatibility complexes (pMHC). The data is provided in five partitions containing one-hot encoded sequences, pre-computed energy terms and corresponding labels.

We have chosen two different approaches for this classification task that are based on the sequence representation. The first representation uses the provided one-hot encoded vector representations for the protein sequences while the other representation is made via an ESM-1b transformer. 

#### ESM-1b embedded model 
In order to run the model with the ESM-1b embedded sequences, a few steps have to be done prior. The data_prep.py, dim_reduction.py and esm_network.py scripts all belong to the ESM-1b embedded model. Their application is described below.

The data_prep.py script is firstly used to split the given data into sequences, local and global energies and subsequently converting the one-hot encoded sequence into amino acid sequence strings. Only the TCR-peptide part of the sequences are converted and saved. The script writes out three fasta files; a training, a validation and a test fasta.

The ESM-1b position embeddings can then be extracted for each amino acid of the sequences with the extract.py script taking the fasta files as input. As this requires much computational power, this sould be done with a high performance computer. The script is run using the following command that extracts both mean and per token embeddings from the last layer of the ESM-1b transformer. 

```
python3 extract.py esm1b_t33_650M_UR50S train_sequences.fasta train_seq_emb_esm1b/ --repr_layers 33 --include mean per_tok
```
Running the code above will result in a directory with a .pt file for each protein complex. 

With the dim_reduction.py script, the dimensionalities of the per token representations can be reduced with a Principal Component Analysis (PCA) that reduces the 1280 features to 100 features to minimize memory usage.

The ESM-1b embedded model can then be run in the esm_network.py script. Here corresponding sequences, local and global energy terms are appended and following the 4 partitions are concatenated as preparation for cross-validation. A parallel network is then defined where the sequence is run through two CNNs and a bi-LSTM, while the global energy terms are run through two linear layers. The output from these are concatenated in a linear layer. The network is then trained using 4-fold cross validation and lastly tested on a fifth partition. 


#### One-hot embedded model 
The onehot_network.py script runs the same network and training structure as in the esm_network.py script but without splitting the features. 

See "Model_with_onehot.ipynb" for model trained on one-hot encoded sequences. The code can be executed by clicking on the icon below:

[<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/JulieMariaJ/TCR-pMHC/blob/main/Model_with_onehot.ipynb)
