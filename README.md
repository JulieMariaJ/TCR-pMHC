# TCR-pMHC
### Prediction of TCR-pMHC Interactions Using Molecular Modeling and Recurrent Networks

This repository contains project data and code for prediction of binding between T-cell receptors (TCR) and peptides presented on major histocompatibility complexes (pMHC). The data is provided in five partitions containing one-hot encoded sequences, pre-computed energy terms and corresponding labels.

The data_prep.py script is firstly used to split the given data into sequences, local and global energies and subsequently converting the one-hot encoded sequence into amino acid sequence strings. Only the TCR-peptide part of the sequences are converted and saved. The script writes out three fasta files; a training, a validation and a test fasta.

The ESM-1b position embeddings can then be extracted for each amino acid of the sequences with the extract.py script taking the fasta files as input. As this requires much computational power, this can for instance be done in  

'''
python3 extract.py esm1b_t33_650M_UR50S train_sequences.fasta train_seq_emb_esm1b/ --repr_layers 33 --include mean per_tok
'''
