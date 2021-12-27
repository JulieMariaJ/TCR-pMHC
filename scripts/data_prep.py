import numpy as np 
import torch
from numpy import savez_compressed

########## functions ##########

# function converting one-hot vektor to sequence string 
def onehot_to_seq(single_seq):
    """ Converting one-hot encoded vector to amino acid sequence string """
    AA = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

    one_hot = []
    for i in range(len(single_seq)):
        one_hot.append(single_seq[i][:20])

    seq = ""
    for i in range(len(one_hot)):
        int_list = one_hot[i].astype(int)
        try: 
          ix = list(int_list).index(1)
          seq += AA[ix]
        except: 
          seq += '-'
    return seq

# function splitting data into sequence, local and global energies 
def split_data(X):
    """ split X into sequence, local and global energies """

    X_seq = []
    X_localE = []
    X_globalE = []

    for sample in X:
        X_seq.append(onehot_to_seq(sample)) # convert one-hot to sequence

        locals = []
        globals = []
        count = 1

        for i in range(len(sample)):
            if count >= 179:
                locals.append(sample[i][20:27])
                globals.append(sample[i][27:])
            count += 1

        X_localE.append(locals)
        X_globalE.append(globals)

    return X_seq, X_localE, X_globalE

########## device ##########

# cuda 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device (CPU/GPU):", device)

########## load data ##########

#define your path
PATH = 'drive/My Drive/Deep learning project/data/'

# load data 
P1input = np.load(PATH+'P1_input.npz')
P1label = np.load(PATH+'P1_labels.npz')
P2input = np.load(PATH+'P2_input.npz')
P2label = np.load(PATH+'P2_labels.npz')
P3input = np.load(PATH+'P3_input.npz')
P3label = np.load(PATH+'P3_labels.npz')
P4input = np.load(PATH+'P4_input.npz')
P4label = np.load(PATH+'P4_labels.npz')
P5input = np.load(PATH+'P5_input.npz')
P5label = np.load(PATH+'P5_labels.npz')

# convert to list 
P1_input_list = list(P1input.values())[0]
P2_input_list = list(P2input.values())[0]
P3_input_list = list(P3input.values())[0]
P4_input_list = list(P4input.values())[0]
P5_input_list = list(P5input.values())[0]

P1_label_list = list(P1label.values())[0]
P2_label_list = list(P2label.values())[0]
P3_label_list = list(P3label.values())[0]
P4_label_list = list(P4label.values())[0]
P5_label_list = list(P5label.values())[0]

# define train, val and test
X_train = np.concatenate([P1_input_list, P2_input_list, P3_input_list])
y_train = np.concatenate([P1_label_list, P2_label_list, P3_label_list])

X_val = P4_input_list
y_val = P4_label_list

X_test = P5_input_list
y_test = P5_label_list

########## split data into seq, local and global energies ##########

Xtrain, X_train_localE, X_train_globalE = split_data(X_train)
Xval, X_val_localE, X_val_globalE = split_data(X_val)
Xtest, X_test_localE, X_test_globalE = split_data(X_test)

# save energies in .npz files 
savez_compressed('train_local_energies.npz', X_train_localE)
savez_compressed('train_global_energies.npz', X_train_globalE)
savez_compressed('val_local_energies.npz', X_val_localE)
savez_compressed('val_global_energies.npz', X_val_globalE)
savez_compressed('test_local_energies.npz', X_test_localE)
savez_compressed('test_global_energies.npz', X_test_globalE)

########## save sequences in fasta files ##########

# write fasta file 
trainfile = open("train_pTCR_seq.fasta", "w")
valfile = open("val_pTCR_seq.fasta", "w")
testfile = open("test_pTCR_seq.fasta", "w")

for i in range(len(Xtrain)):
  trainfile.write(">" + "protein{0}".format(i) + "\n" + Xtrain[i][178:] + "\n") # discard MHC

for i in range(len(Xval)):
  valfile.write(">" + "protein{0}".format(i) + "\n" + Xval[i][178:] + "\n") # discard MHC

for i in range(len(Xtest)):
  testfile.write(">" + "protein{0}".format(i) + "\n" + Xtest[i][178:] + "\n") # discard MHC

trainfile.close()
valfile.close()
testfile.close()
