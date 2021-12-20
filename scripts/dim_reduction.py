import numpy as np
import torch
from numpy import savez_compressed
from sklearn.decomposition import PCA

########## load embeddings ##########

#define your path 
PATH = "/zhome/77/7/127802/data/"

train_FASTA_PATH = PATH + "train_pTCR_seq.fasta"
train_EMB_PATH = PATH + "train_pTCR_emb_esm1b/"

val_FASTA_PATH = PATH + "val_pTCR_seq.fasta"
val_EMB_PATH = PATH + "val_pTCR_emb_esm1b/"

test_FASTA_PATH = PATH + "test_sequences.fasta"
test_EMB_PATH = PATH + "test_pTCR_emb_esm1b/"

########## extract representations ##########

Xs_mean = []
Xs_tok = []
val_Xs_mean = []
val_Xs_tok = []
test_X_mean = []
test_X_tok = []
count = 1
for header, _seq in esm.data.read_fasta(train_FASTA_PATH):
    if count <= 1532:
        if count <= 1207:
            fn_test = f'{test_EMB_PATH}/{header[1:]}.pt' # test embeddings
            embs_test = torch.load(fn_test)
            test_X_mean.append(embs_test['mean_representations'][33])
            test_X_tok.append(embs_test['representations'][33])

        fn_val = f'{val_EMB_PATH}/{header[1:]}.pt' # val embeddings
        embs_val = torch.load(fn_val)
        val_Xs_mean.append(embs_val['mean_representations'][33])
        val_Xs_tok.append(embs_val['representations'][33])

    fn_train = f'{train_EMB_PATH}/{header[1:]}.pt' # train embeddings
    embs_train = torch.load(fn_train)
    Xs_mean.append(embs_train['mean_representations'][33])
    Xs_tok.append(embs_train['representations'][33])    

    count += 1

Xs_mean_train = torch.stack(Xs_mean, dim=0).numpy()
Xs_tok_train = torch.stack(Xs_tok, dim=0).numpy()
Xs_mean_val = torch.stack(val_Xs_mean, dim=0).numpy()
Xs_tok_val = torch.stack(val_Xs_tok, dim=0).numpy()
Xs_mean_test = torch.stack(test_X_mean, dim=0).numpy()
Xs_tok_test = torch.stack(test_X_tok, dim=0).numpy()

# save mean representations
savez_compressed(PATH+"Xtrain_meanRep.npz", Xs_mean_train)
savez_compressed(PATH+"Xval_meanRep.npz", Xs_mean_val)
savez_compressed(PATH+"Xtest_meanReo.npz", Xs_mean_test)

########## PCA ##########

# do PCA on train set 
Xs_tok_pca = np.reshape(Xs_tok_train,(Xs_tok_train.shape[0]*Xs_tok_train.shape[1], Xs_tok_train.shape[2]))
pca = PCA(100)
Xs_train_pca = pca.fit_transform(Xs_tok_pca)
Xs_after_pca = np.reshape(Xs_train_pca, (Xs_tok_train.shape[0], Xs_tok_train.shape[1], 100)) 
#print(pca.explained_variance_ratio_.cumsum())

# fit PCA on val set
Xs_val_2d = np.reshape(Xs_tok_val, (Xs_tok_val.shape[0]*Xs_tok_val.shape[1], Xs_tok_val.shape[2]))
Xs_val_pca = pca.transform(Xs_val_2d)
Xs_val_pca = np.reshape(Xs_val_pca, (Xs_tok_val.shape[0], Xs_tok_val.shape[1], 100))

# fit PCA on test set 
Xs_test_2d = np.reshape(Xs_tok_test, (Xs_tok_test.shape[0]*Xs_tok_test.shape[1], Xs_tok_test.shape[2]))
Xs_test_pca = pca.transform(Xs_test_2d)
Xs_test_pca = np.reshape(Xs_test_pca, (Xs_tok_test.shape[0], Xs_tok_test.shape[1], 100))

#save pca representations
savez_compressed(PATH+"train_emb_pca.npz", Xs_after_pca)
savez_compressed(PATH+"val_emb_pca.npz", Xs_val_pca)
savez_compressed(PATH+"test_emb_pca.npz", Xs_test_pca)
