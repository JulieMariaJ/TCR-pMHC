import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_curve, confusion_matrix
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_curve
from torch.nn.functional import relu, sigmoid
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

########## load data ##########

#define your path 
PATH = "drive/My Drive/Deep learning project/data/"

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

########## define data ##########

# collect input and label 
trainset = []
for i in range(len(X_train)):
  trainset.append([np.transpose(X_train[i]), y_train[i]])

valset = []
for i in range(len(X_val)):
  valset.append([np.transpose(X_val[i]), y_val[i]])

# prepare data for CV 
X = np.concatenate([P1_input_list, P2_input_list, P3_input_list, P4_input_list])
y = np.concatenate([P1_label_list, P2_label_list, P3_label_list, P4_label_list])

dataset = []
for i in range(len(y)):
  dataset.append([np.transpose(X[i]), y[i]])

dataset_split = []
dataset_split.append(dataset[:1526])
dataset_split.append(dataset[1526:1526+1168])
dataset_split.append(dataset[1526+1168:1526+1168+1480])
dataset_split.append(dataset[1526+1168+1480:])

########## define baseline model (simple CNN) ##########

channels = X_train.shape[2]
dim_size = X_train.shape[1]
num_classes = 1

num_filters1 = 100
kernel_size_conv1 = 3 
stride_conv1 = 2
padding_conv1 = 1

# define network
def compute_conv_dim(dim_size, kernel_size, padding_size, stride_size):
    return int((dim_size - kernel_size + 2 * padding_size) / stride_size + 1)

class baseline_Net(nn.Module):
    def __init__(self,  num_classes):
        super(baseline_Net, self).__init__()  
        self.bn0 = nn.BatchNorm1d(channels)   
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=num_filters1, kernel_size=kernel_size_conv1, stride=stride_conv1, padding=padding_conv1)
        self.conv_bn1 = nn.BatchNorm1d(num_filters1) 
        self.conv_out_dim1 = compute_conv_dim(dim_size, kernel_size_conv1, padding_conv1, stride_conv1)
        
        self.fc1 = nn.Linear(self.conv_out_dim1*num_filters1, num_classes)

    def forward(self, x):  
        x = self.bn0(x)    
        x = relu(self.conv1(x))
        x = self.conv_bn1(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc1(x))
        return x

baseline_net = baseline_Net(num_classes).to(device)
print(baseline_net)

########## define network for one-hot embedding ##########

# hyper parameters
channels = x_train.shape[2]
dim_size = x_train.shape[1]
num_classes = 1

num_filters1 = 100
kernel_size_conv1 = 3 
stride_conv1 = 2
padding_conv1 = 1

num_filters2 = 100
kernel_size_conv2 = 3 
stride_conv2 = 2
padding_conv2 = 1

l1_num = 10
hidden_size = 26


# define network
def compute_conv_dim(dim_size, kernel_size, padding_size, stride_size):
    return int((dim_size - kernel_size + 2 * padding_size) / stride_size + 1)

def pool_dim(dim_size, kernel_size, padding_size, stride_size):
    return int((dim_size - 2)/2 + 1 )

class One_Hot_Net(nn.Module):
    def __init__(self,  num_classes):
        super(One_Hot_Net, self).__init__()  
        self.bn0 = nn.BatchNorm1d(channels)   
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=num_filters1, kernel_size=kernel_size_conv1, stride=stride_conv1, padding=padding_conv1)
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        self.conv_bn1 = nn.BatchNorm1d(num_filters1) 
        self.conv_out_dim1 = compute_conv_dim(dim_size, kernel_size_conv1, padding_conv1, stride_conv1)

        self.conv2 = nn.Conv1d(in_channels=num_filters1, out_channels=num_filters2, kernel_size=kernel_size_conv2, stride=stride_conv2, padding=padding_conv2)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)
        self.conv_bn2 = nn.BatchNorm1d(num_filters2) 
        self.conv_out_dim2 = compute_conv_dim(self.conv_out_dim1, kernel_size_conv2, padding_conv2, stride_conv2)
        
        self.rnn = nn.LSTM(input_size=num_filters2, hidden_size=hidden_size, num_layers=3, batch_first=True, bidirectional = True)

        self.fc1 = nn.Linear(2*hidden_size, num_classes)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        
        #dropout
        self.dropout_cnn = nn.Dropout(0.2)
        self.dropout_linear = nn.Dropout(0.5) 

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):  
        x = self.bn0(x)    
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv_bn1(x)
        x = self.dropout_cnn(x)

        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv_bn2(x)
        x = self.dropout_cnn(x)

        x = x.transpose_(2, 1)
        x, (h, c) = self.rnn(x)
        cat = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        cat = self.dropout_cnn(cat)

        x = torch.sigmoid(self.fc1(cat))
        return x

One_Hot_net = One_Hot_Net(num_classes).to(device)
print(One_Hot_net)

########## loss and optimizer ##########

import torch.optim as optim

criterion = nn.BCELoss()  
optimizer = optim.Adam(One_Hot_net.parameters(), lr=0.001, weight_decay=1e-8)

########## reset weights function ##########

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    #print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

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

    # Define data loaders for training and testing data in this fold
    trainloader_fold = torch.utils.data.DataLoader(
                        train_set_flat, 
                        batch_size=64, shuffle = True)
    
    x_train, y_train = next(iter(trainloader_fold))
    
    channels = x_train.shape[2]
    dim_size = x_train.shape[1]

    valloader_fold = torch.utils.data.DataLoader(
                        dataset_split[fold],
                        batch_size=64, shuffle = True)
    
    One_Hot_net.apply(reset_weights)
    
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
        One_Hot_net.train()
        train_targets, train_preds, train_preds_sig = [], [], []
        for i, (data, target) in enumerate(trainloader_fold):
            # get the inputs
            x_train = data.float().detach().requires_grad_(True).cuda()
            labels_train = torch.tensor(np.array(target), dtype=torch.float).unsqueeze(1).cuda()

            optimizer.zero_grad()

            output = One_Hot_net(x_train)

            train_batch_loss = criterion(output, labels_train)
            train_batch_loss.backward()
            optimizer.step()
            
            preds = np.round(output.detach().cpu())
            preds_sig = output.detach().cpu()
            train_targets += list(np.array(labels_train.cpu()))
            train_preds += list(preds.numpy().flatten())
            train_preds_sig += list(preds_sig.numpy().flatten())
            running_loss += train_batch_loss.item()

        losses.append(running_loss / len(trainloader_fold.dataset))

        # evaluate
        One_Hot_net.eval()
        val_targets, val_preds, val_preds_sig = [], [], []
        with torch.no_grad():
            for i, (data, target) in enumerate(valloader_fold):
                # get the inputs
                x_val = data.float().detach().requires_grad_(True).cuda()
                labels_val = torch.tensor(np.array(target), dtype=torch.float).unsqueeze(1).cuda()
                
                output = One_Hot_net(x_val)
                
                val_batch_loss = criterion(output, labels_val)

                val_predicts_sig = output.detach().cpu()
                val_predicts = np.round(output.detach().cpu())
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

        # Early stopping
        if (roc_auc_val / len(valloader_fold.dataset)) > max_val_auc:   #(val_loss / len(valloader_fold.dataset)) < min_val_loss:
            print("Model found at epoch:", epoch)  # save model/predictions with lowest loss 
            best_net = One_Hot_net
            best_val_targets = val_targets
            best_val_preds = val_preds
            best_val_preds_sig = val_preds_sig
            best_train_targets = train_targets
            best_train_preds = train_preds
            best_train_preds_sig = train_preds_sig
            no_epoch_improve = 0
            max_val_auc = (roc_auc_val / len(valloader_fold.dataset))     #min_val_loss = (val_loss / len(valloader_fold.dataset))
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

########## define test data ##########

testset = []
for i in range(len(X_test)):
  testset.append([np.transpose(X_test[i]), y_test[i]])

# Data loader 
batchsize = 64
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=True)

########## test network ##########

#### concatenate models #####

all_test_preds, all_test_targs = [], []
all_test_preds_sig = []
all_test_preds_mcc = []

for i in range(4):
    cur_net = torch.load('best_net'+str(i)+'.pth')

    test_preds, test_targs = [], []
    test_preds_sig = []
    test_preds_mcc = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(testloader):
            x_test = data.float().detach().requires_grad_(True).cuda()
            labels_test = torch.tensor(np.array(target), dtype=torch.float).unsqueeze(1).cuda()
          
            output = cur_net(x_test)

            test_batch_loss = criterion(output, labels_test)

            test_predicts = np.round(output.detach().cpu())
            test_predicts_sig = output.detach().cpu()
            test_pred_mcc = mcc_threshold(output.detach().cpu())
            test_targs += list(np.array(labels_test.cpu()))
            test_preds += list(test_predicts.numpy().flatten())
            test_preds_sig += list(test_predicts_sig.numpy().flatten())
            test_preds_mcc += list(test_pred_mcc)

    all_test_preds.append(test_preds)
    all_test_preds_sig.append(test_preds_sig)
    all_test_preds_mcc.append(test_preds_mcc)
    all_test_targs.append(test_targs)  

all_test_preds = sum(all_test_preds, [])
all_test_preds_sig = sum(all_test_preds_sig, [])
all_test_preds_mcc = sum(all_test_preds_mcc, [])
all_test_targs = sum(all_test_targs, [])

fpr_test, tpr_test, threshold_test = metrics.roc_curve(all_test_targs, all_test_preds_sig)
roc_auc_test = metrics.auc(fpr_test, tpr_test)
print("Test AUC:", roc_auc_test)
print("Test MCC:", matthews_corrcoef(all_test_targs, all_test_preds))

########## Performance ##########

c = 3

#Loss curve 
epoch = np.arange(len(master_train_losses[c]))
plt.figure()
plt.plot(epoch, master_train_losses[c], 'r', epoch, master_val_losses[c], 'b')
plt.legend(['Train loss','Validation loss'])
plt.xlabel('Epochs'), plt.ylabel('Loss')
plt.show

#Loss curve based on AUC 
epoch = np.arange(len(master_train_aucs[c]))
plt.figure()
plt.plot(epoch, master_train_aucs[c], 'r', epoch, master_val_aucs[c], 'b')
plt.legend(['Train AUC','Validation AUC'])
plt.xlabel('Epochs'), plt.ylabel('AUC')
plt.show()

# AUC plot 
def plot_roc(targets, predictions, filename):
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

### concatenated ###
best_train_targets_list_cat = sum(best_train_targets_list, [])
best_train_preds_list_cat = sum(best_train_preds_list, [])
best_val_targets_list_cat = sum(best_val_targets_list, [])
best_val_preds_list_cat = sum(best_val_preds_list, [])

best_train_preds_sig_list_cat = sum(best_train_preds_sig_list, [])
best_val_preds_sig_list_cat = sum(best_val_preds_sig_list, [])

plot_roc(best_train_targets_list_cat, best_train_preds_sig_list_cat, "train_AUC")
plot_roc(best_val_targets_list_cat, best_val_preds_sig_list_cat, "val_AUC")

# performance score for concatenated results

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
