import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_curve, confusion_matrix
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_curve
from torch.autograd import Variable
from torch.nn.functional import relu, sigmoid
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

########## load data ##########

#define your path 
PATH = "drive/My Drive/Deep learning project/data/"

# load y-values 
y_train1 = np.load(PATH+"P1_labels.npz")
y_train2 = np.load(PATH+"P2_labels.npz")
y_train3 = np.load(PATH+"P3_labels.npz")
y_val = np.load(PATH+"P4_labels.npz")

y_train1 = list(y_train1.values())[0]
y_train2 = list(y_train2.values())[0]
y_train3 = list(y_train3.values())[0]
y_val = list(y_val.values())[0]

y_train = np.concatenate([y_train1, y_train2, y_train3])

# load embeddings 
X_train_emb = np.load(PATH+"train_emb_pca.npz")
X_val_emb = np.load(PATH+"val_emb_pca.npz")

X_train_emb = list(X_train_emb.values())[0]
X_val_emb = list(X_val_emb.values())[0]

# load local energies 
X_train_LE = np.load(PATH+"train_local_energies.npz")
X_val_LE = np.load(PATH+"val_local_energies.npz")

X_train_LE = list(X_train_LE.values())[0]
X_val_LE = list(X_val_LE.values())[0]

# load global energies 
X_train_GE = np.load(PATH+"train_global_energies.npz")
X_val_GE = np.load(PATH+"val_global_energies.npz")

X_train_GE = list(X_train_GE.values())[0]
X_val_GE = list(X_val_GE.values())[0]

########## concatenate train and val data to make 4-fold CV ##########

dataset = []
for i in range(len(y_train)):
  dataset.append([np.transpose(X_train_emb[i]), np.transpose(X_train_LE[i]), np.transpose(X_train_GE[i]), y_train[i]])

for i in range(len(y_val)):
  dataset.append([np.transpose(X_val_emb[i]), np.transpose(X_val_LE[i]), np.transpose(X_val_GE[i]), y_val[i]])

########## define device ##########

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device (CPU/GPU):", device)

########## define network ##########

# hyper parameters
x_train_emb = torch.empty((1,100,242), dtype=torch.int32, device = 'cuda')
x_train_LE = torch.empty((1,7,242), dtype=torch.int32, device = 'cuda')
x_train_GE = torch.empty((1,27,242), dtype=torch.int32, device = 'cuda')

channels = x_train_emb.shape[1]
dim_size = x_train_emb.shape[2]
num_classes = 1

num_filters1 = 200
kernel_size_conv1 = 2
stride_conv1 = 2
padding_conv1 = 1

num_filters2 = 200
kernel_size_conv2 = 2
stride_conv2 = 2
padding_conv2 = 1

channels_GE = x_train_GE.shape[1]
dim_size_GE = x_train_GE.shape[2]

num_l1 = 200
hidden_size = 50
num_l2 = 100

def compute_conv_dim(dim_size, kernel_size, padding_size, stride_size):
    """ calculate CNN out dimension """
    return int((dim_size - kernel_size + 2 * padding_size) / stride_size + 1)

class Net(nn.Module):
    def __init__(self,  num_classes):
        super(Net, self).__init__()
        self.bn0 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=num_filters1, kernel_size=kernel_size_conv1, stride=stride_conv1, padding=padding_conv1)
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        self.conv_bn1 = nn.BatchNorm1d(num_filters1)
        self.conv_out_dim1 = compute_conv_dim(dim_size, kernel_size_conv1, padding_conv1, stride_conv1)

        self.conv2 = nn.Conv1d(in_channels=num_filters1, out_channels=num_filters2, kernel_size=kernel_size_conv2, stride=stride_conv2, padding=padding_conv2)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)
        self.conv_bn2 = nn.BatchNorm1d(num_filters2)
        self.conv_out_dim2 = compute_conv_dim(self.conv_out_dim1, kernel_size_conv2, padding_conv2, stride_conv2)

        self.rnn = nn.LSTM(input_size=num_filters1, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional = True)
       
        self.l1_GE = nn.Linear(in_features = x_train_GE.shape[1], out_features = num_l1)
        self.l1_bn4 = nn.BatchNorm1d(num_l1)
        self.l2_GE = nn.Linear(in_features = num_l1, out_features = num_l2)
        self.l2_bn4 = nn.BatchNorm1d(num_l2)

        self.features_cat_size = (2 * hidden_size) + num_l2 

        self.l_out = nn.Linear(self.features_cat_size, num_classes)
        torch.nn.init.xavier_uniform_(self.l_out.weight)

        #dropout
        self.dropout_cnn_lstm = nn.Dropout(0.2)
        self.dropout_linear = nn.Dropout(0.5)

        # maxpool
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x_emb, x_LE, x_GE):
        features = []
        out = {}
        
        # conv 1
        x_emb = self.bn0(x_emb)
        x_emb = self.pool(F.relu(self.conv1(x_emb)))
        x_emb = self.conv_bn1(x_emb)
        x_emb = self.dropout_cnn_lstm(x_emb)

         # conv 2
        x_emb = self.pool(F.relu(self.conv2(x_emb)))
        x_emb = self.conv_bn2(x_emb)
        x_emb = self.dropout_cnn_lstm(x_emb)
        
        # bi-LSTM
        x_emb = x_emb.transpose_(2, 1)
        x_emb, (h, c) = self.rnn(x_emb)
        cat = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        cat = self.dropout_cnn_lstm(cat)
        features.append(cat)

        # linear (global E)
        x_GE = x_GE.view(x_GE.shape[0], -1)
        x_GE = x_GE[:,:27]
        
        x_GE = relu(self.l1_GE(x_GE))
        x_GE = self.l1_bn4(x_GE)
        x_GE = self.dropout_linear(x_GE)
        x_GE = relu(self.l2_GE(x_GE))
        x_GE = self.l2_bn4(x_GE)
        x_GE = self.dropout_linear(x_GE)
        features.append(x_GE)

        # linear out
        features_out = torch.cat(features, dim=1)
        features_out = self.l_out(features_out)
        out['out'] = torch.sigmoid(features_out)

        return out

net = Net(num_classes).to(device)
print(net)

########## loss function and optimizer ##########

criterion = nn.BCELoss()  
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-8)

########## reseting weights function ##########
def reset_weights(m):
  """ reset weights """
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    #print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

########## Use premade partitioning for CV ##########
dataset_split = []
dataset_split.append(dataset[:1526])
dataset_split.append(dataset[1526:1526+1168])
dataset_split.append(dataset[1526+1168:1526+1168+1480])
dataset_split.append(dataset[1526+1168+1480:])

########## Train network ##########

num_epoch = 100

# For fold results
train_results = {}
val_results = {}
best_nets, best_val_targets_list, best_val_preds_list, best_train_targets_list, best_train_preds_list = [], [], [], [], []
best_train_preds_sig_list, best_val_preds_sig_list = [], []
master_train_losses, master_val_losses = [], []
master_train_aucs, master_val_aucs = [], []

for fold, sample in enumerate(dataset_split):  
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')

    train_set = dataset_split[:fold]+dataset_split[fold+1:]
    train_set_flat = [item for sublist in train_set for item in sublist]

    # Define data loaders for training and validation data in this fold
    trainloader_fold = torch.utils.data.DataLoader(
                        train_set_flat, 
                        batch_size=64,
                        shuffle=True)
    
    x_train_emb, x_train_LE, x_train_GE, y_train = next(iter(trainloader_fold))
    
    channels = x_train_emb.shape[1]
    dim_size = x_train_emb.shape[2]
    channels_LE = x_train_LE.shape[1]
    dim_size_LE = x_train_LE.shape[2]

    valloader_fold = torch.utils.data.DataLoader(
                        dataset_split[fold],
                        batch_size=64,
                        shuffle=True)
    
    net.apply(reset_weights) # reset weights 
    
    # initialize
    no_epoch_improve = 0
    max_val_auc = 0

    train_loss = []
    losses = []
    val_losses = []
    train_acc = []
    valid_acc = []
    minimum_val_loss = 1000

    train_aucs = []
    val_aucs = []

    for epoch in range(num_epoch):  
        print('epoch number:', epoch)
        running_loss = 0.0
        val_loss = 0.0

        # train network
        net.train()
        train_targets, train_preds, train_preds_sig = [], [], []
        for i, (emb, LE, GE, target) in enumerate(trainloader_fold):
            # get the inputs
            emb_train = emb.float().detach().requires_grad_(True).cuda()
            LE_train = LE.float().detach().requires_grad_(True).cuda()
            GE_train = GE.float().detach().requires_grad_(True).cuda()
            labels_train = torch.tensor(np.array(target), dtype=torch.float).unsqueeze(1).cuda()

            optimizer.zero_grad()

            output = net(emb_train, LE_train, GE_train)

            train_batch_loss = criterion(output['out'], labels_train)
            train_batch_loss.backward()
            optimizer.step()
            
            preds = np.round(output['out'].detach().cpu())
            preds_sig = output['out'].detach().cpu()
            train_targets += list(np.array(labels_train.cpu()))
            train_preds += list(preds.numpy().flatten())
            train_preds_sig += list(preds_sig.numpy().flatten())
            running_loss += train_batch_loss.item()


        losses.append(running_loss / len(trainloader_fold.dataset))

        # evaluate
        net.eval()
        val_targets, val_preds, val_preds_sig = [], [], []
        with torch.no_grad():
            for i, (emb, LE, GE, target) in enumerate(valloader_fold):
                # get the inputs
                emb_val = emb.float().detach().requires_grad_(True).cuda()
                LE_val = LE.float().detach().requires_grad_(True).cuda()
                GE_val = GE.float().detach().requires_grad_(True).cuda()
                labels_val = torch.tensor(np.array(target), dtype=torch.float).unsqueeze(1).cuda()
                
                output = net(emb_val, LE_val, GE_val)
                
                val_batch_loss = criterion(output['out'], labels_val)

                val_predicts_sig = output['out'].detach().cpu()
                val_predicts = np.round(output['out'].detach().cpu())
                val_targets += list(np.array(labels_val.cpu()))
                val_preds += list(val_predicts.numpy().flatten())
                val_preds_sig += list(val_predicts_sig.numpy().flatten())
                val_loss += val_batch_loss.item()
        
        val_losses.append(val_loss / len(valloader_fold.dataset))
        print("validation loss:",val_loss/len(valloader_fold.dataset))
 
        train_acc_cur = accuracy_score(train_targets, train_preds)
        valid_acc_cur = accuracy_score(val_targets, val_preds)
        
        train_acc.append(train_acc_cur)
        valid_acc.append(valid_acc_cur)

        # AUC
        fpr_train, tpr_train, threshold_train = metrics.roc_curve(train_targets, train_preds_sig)
        roc_auc_train = metrics.auc(fpr_train, tpr_train)
        fpr_val, tpr_val, threshold_val = metrics.roc_curve(val_targets, val_preds_sig)
        roc_auc_val = metrics.auc(fpr_val, tpr_val)
        train_aucs.append(roc_auc_train)
        val_aucs.append(roc_auc_val)

        # Early stopping based on validation AUC
        if (roc_auc_val / len(valloader_fold.dataset)) > max_val_auc:  
            print("Model found at epoch:", epoch)  # save model/predictions with lowest loss 
            best_net = net
            best_val_targets = val_targets
            best_val_preds = val_preds
            best_val_preds_sig = val_preds_sig
            best_train_targets = train_targets
            best_train_preds = train_preds
            best_train_preds_sig = train_preds_sig
            no_epoch_improve = 0
            max_val_auc = (roc_auc_val / len(valloader_fold.dataset))  
        else:
            no_epoch_improve +=1
        if no_epoch_improve == 10:
            print("Early stopping\n")
            break

    #save intermediate models 
    best_nets.append(best_net)
    torch.save(best_net, 'best_net{0}.pth'.format(fold))
    best_val_targets_list.append(best_val_targets)
    best_val_preds_list.append(best_val_preds)
    best_train_targets_list.append(best_train_targets)
    best_train_preds_list.append(best_train_preds)
    best_train_preds_sig_list.append(best_train_preds_sig)
    best_val_preds_sig_list.append(best_val_preds_sig)

    # save losses per fold
    master_train_losses.append(losses)
    master_val_losses.append(val_losses)
    
    # save all AUCs in fold
    master_train_aucs.append(train_aucs)
    master_val_aucs.append(val_aucs)

    # best AUC per fold 
    fpr_train, tpr_train, threshold_train = metrics.roc_curve(best_train_targets, best_train_preds_sig)
    roc_auc_train = metrics.auc(fpr_train, tpr_train)
    train_results[fold + 1] = roc_auc_train
    
    fpr_val, tpr_val, threshold_val = metrics.roc_curve(best_val_targets, best_val_preds_sig)
    roc_auc_val = metrics.auc(fpr_val, tpr_val)
    val_results[fold + 1] = roc_auc_val
  
# display AUC per fold 
sum_train = 0.0
sum_val = 0.0
for key, value in train_results.items():
  print(f'Fold {key}: {value} %')
  sum_train += value
print(f'Average: {sum_train/len(train_results.items())} %')
for key, value in val_results.items():
  print(f'Fold {key}: {value} %')
  sum_val += value
print(f'Average: {sum_val/len(val_results.items())} %')

########## Test network ##########

# load test set 
P5input = np.load('drive/My Drive/Deep learning project/data/P5_input.npz')
P5label = np.load('drive/My Drive/Deep learning project/data/P5_labels.npz')
X_test = list(P5input.values())[0]
y_test = list(P5label.values())[0]

# load test embeddings 
X_test_emb = np.load("drive/My Drive/Deep learning project/data/test_emb_pca.npz")
X_test_emb = list(X_test_emb.values())[0]

# get test energies 
X_test_localE = [] 
X_test_globalE = [] 

for sample in X_test:
  locals = []
  globals = []
  count = 1

  for i in range(len(sample)):
    if count >= 179:
      locals.append(sample[i][20:27])
      globals.append(sample[i][27:])
    count += 1

  X_test_localE.append(locals)
  X_test_globalE.append(globals)

X_test_localE = np.array(X_test_localE)
X_test_globalE = np.array(X_test_globalE)

# collect test data
testset = []
for i in range(len(X_test)):
  testset.append([np.transpose(X_test_emb[i]), np.transpose(X_test_localE[i]), np.transpose(X_test_globalE[i]), y_test[i]])

batchsize = 64
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=True)

########## concatenate best models per fold ##########

all_test_preds, all_test_targs = [], []
all_test_preds_sig = []

for i in range(4):
    cur_net = torch.load('best_net'+str(i)+'.pth')

    test_preds, test_targs = [], []
    test_preds_sig = []
    with torch.no_grad():
        for batch_idx, (emb, LE, GE, target) in enumerate(testloader):
            emb_test = emb.float().detach().requires_grad_(True).cuda()
            LE_test = LE.float().detach().requires_grad_(True).cuda()
            GE_test = GE.float().detach().requires_grad_(True).cuda()
            labels_test = torch.tensor(np.array(target), dtype=torch.float).unsqueeze(1).cuda()
          
            output = cur_net(emb_test, LE_test, GE_test)

            test_batch_loss = criterion(output['out'], labels_test)

            test_predicts = np.round(output['out'].detach().cpu())
            test_predicts_sig = output['out'].detach().cpu()
            test_targs += list(np.array(labels_test.cpu()))
            test_preds += list(test_predicts.numpy().flatten())
            test_preds_sig += list(test_predicts_sig.numpy().flatten())

    all_test_preds.append(test_preds)
    all_test_preds_sig.append(test_preds_sig)
    all_test_targs.append(test_targs)  

all_test_preds = sum(all_test_preds, [])
all_test_preds_sig = sum(all_test_preds_sig, [])
all_test_targs = sum(all_test_targs, [])

fpr_test, tpr_test, threshold_test = metrics.roc_curve(all_test_targs, all_test_preds_sig)
roc_auc_test = metrics.auc(fpr_test, tpr_test)
print("Test AUC:", roc_auc_test)
print("Test MCC:", matthews_corrcoef(all_test_targs, all_test_preds))

########## Performance ##########

# plot learning curve based on loss
c = 0
epoch = np.arange(len(master_train_losses[c]))
plt.figure()
plt.plot(epoch, master_train_losses[c], 'r', epoch, master_val_losses[c], 'b')
plt.legend(['Train loss','Validation loss'])
plt.xlabel('Epochs'), plt.ylabel('Loss')
plt.show()

# plot learning curve based on AUC
epoch = np.arange(len(master_train_aucs[c]))
plt.figure()
plt.plot(epoch, master_train_aucs[c], 'r', epoch, master_val_aucs[c], 'b')
plt.legend(['Train AUC','Validation AUC'])
plt.xlabel('Epochs'), plt.ylabel('AUC')
plt.show()

# plot AUC 
def plot_roc(targets, predictions, filename):
    """ function for plitting AUC """
    # ROC
    fpr, tpr, threshold = metrics.roc_curve(targets, predictions)
    roc_auc = metrics.auc(fpr, tpr)

    # plot ROC
    plt.figure()
    plt.title('Validation ROC-curve')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# plot ROC-curve for concatenated results
best_train_targets_list_cat = sum(best_train_targets_list, [])
best_train_preds_list_cat = sum(best_train_preds_list, [])
best_val_targets_list_cat = sum(best_val_targets_list, [])
best_val_preds_list_cat = sum(best_val_preds_list, [])
best_train_preds_sig_list_cat = sum(best_train_preds_sig_list, [])
best_val_preds_sig_list_cat = sum(best_val_preds_sig_list, [])

plot_roc(best_train_targets_list_cat, best_train_preds_list_cat, "train_AUC")
plot_roc(best_val_targets_list_cat, best_val_preds_list_cat, "val_AUC")

plot_roc(best_train_targets_list_cat, best_train_preds_sig_list_cat, "train_AUC")
plot_roc(best_val_targets_list_cat, best_val_preds_sig_list_cat, "val_AUC")

## concatenated performance scores 

# f1 score 
print("F1 score of training", f1_score(best_train_targets_list_cat, best_train_preds_list_cat))
print("F1 score of validation", f1_score(best_val_targets_list_cat, best_val_preds_list_cat))
print("F1 score of test", f1_score(test_targs, test_preds))

# confusion matrix
print("Confusion matrix for training", confusion_matrix(best_train_targets_list_cat, best_train_preds_list_cat))
print("Confusion matrix for validation", confusion_matrix(best_val_targets_list_cat, best_val_preds_list_cat))
print("Confusion matrix for test", confusion_matrix(test_targs, test_preds))

# MCC 
print("MCC for training", matthews_corrcoef(best_train_targets_list_cat, best_train_preds_list_cat))
print("MCC for val", matthews_corrcoef(best_val_targets_list_cat, best_val_preds_list_cat))
