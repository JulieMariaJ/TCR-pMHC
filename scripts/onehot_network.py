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

# load data 
P1input = np.load('drive/My Drive/Deep learning project/data/P1_input.npz')
P1label = np.load('drive/My Drive/Deep learning project/data/P1_labels.npz')
P2input = np.load('drive/My Drive/Deep learning project/data/P2_input.npz')
P2label = np.load('drive/My Drive/Deep learning project/data/P2_labels.npz')
P3input = np.load('drive/My Drive/Deep learning project/data/P3_input.npz')
P3label = np.load('drive/My Drive/Deep learning project/data/P3_labels.npz')
P4input = np.load('drive/My Drive/Deep learning project/data/P4_input.npz')
P4label = np.load('drive/My Drive/Deep learning project/data/P4_labels.npz')
P5input = np.load('drive/My Drive/Deep learning project/data/P5_input.npz')
P5label = np.load('drive/My Drive/Deep learning project/data/P5_labels.npz')

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

# Data loader 
batchsize = 64
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=batchsize, shuffle=True)

########## define baseline model (simple CNN) ##########

channels = x_train.shape[1]
dim_size = x_train.shape[2]
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
channels = x_train.shape[1]
dim_size = x_train.shape[2]
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
        self.conv_bn1 = nn.BatchNorm1d(num_filters1) 
        self.conv_out_dim1 = compute_conv_dim(dim_size, kernel_size_conv1, padding_conv1, stride_conv1)

        self.conv2 = nn.Conv1d(in_channels=num_filters1, out_channels=num_filters2, kernel_size=kernel_size_conv2, stride=stride_conv2, padding=padding_conv2)
        self.conv_bn2 = nn.BatchNorm1d(num_filters2) 
        self.conv_out_dim2 = compute_conv_dim(self.conv_out_dim1, kernel_size_conv2, padding_conv2, stride_conv2)
        
        self.rnn = nn.LSTM(input_size=num_filters2, hidden_size=hidden_size, num_layers=3, batch_first=True, bidirectional = True)

        self.fc1 = nn.Linear(2*hidden_size, num_classes)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):  
        x = self.bn0(x)    
        x = self.pool(F.relu(self.conv1(x)))
        x = self.conv_bn1(x)

        x = self.pool(F.relu(self.conv2(x)))
        x = self.conv_bn2(x)

        x = x.transpose_(2, 1)
        x, (h, c) = self.rnn(x)
        cat = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)

        x = torch.sigmoid(self.fc1(cat))
        return x

One_Hot_net = One_Hot_Net(num_classes).to(device)
print(One_Hot_net)

########## loss and optimizer ##########

import torch.optim as optim

criterion = nn.BCELoss()  
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-8)

########## Train network ##########
num_epoch = 5  

train_loss = []
losses = []
val_losses = []

for epoch in range(num_epoch):  
    running_loss = 0.0
    val_loss = 0.0

    # train network
    net.train() 
    train_targets, train_preds = [], []
    for i, (data, target) in enumerate(trainloader):
        # get the inputs
        inputs_train = data.float().detach().requires_grad_(True).cuda()
        labels_train = torch.tensor(np.array(target), dtype=torch.float).unsqueeze(1).cuda()

        optimizer.zero_grad()

        #output = baseline_net(inputs_train)
        output = One_Hot_net(inputs_train)
        
        train_batch_loss = criterion(output, labels_train)
        train_batch_loss.backward()
        optimizer.step()
         
        preds = np.round(output.detach().cpu())
        train_targets += list(np.array(labels_train.cpu()))
        train_preds += list(preds.data.numpy().flatten())
        running_loss += train_batch_loss.item()
        
    losses.append(running_loss / len(trainloader.dataset))

    # evaluate
    net.eval()
    val_targets, val_preds = [], []
    with torch.no_grad():
        for i, (data, target) in enumerate(valloader):
            # get the inputs
            inputs_val = data.float().detach().cuda()
            labels_val = target.float().detach().unsqueeze(1).cuda() #torch.tensor(np.array(target), dtype=torch.float).unsqueeze(1).cuda()
            
            #output = baseline_net(inputs_val)
            output = One_Hot_net(inputs_val)
            
            val_batch_loss = criterion(output, labels_val)

            val_predicts = np.round(output.detach().cpu())
            val_targets += list(np.array(labels_val.cpu()))
            val_preds += list(val_predicts.data.numpy().flatten())
            val_loss += val_batch_loss.item()
    
    val_losses.append(val_loss / len(valloader.dataset))

    # display train loss per epoch
    print("Training loss:", losses[-1], "\t Validation loss:", val_losses[-1])
    print("Training MCC:", matthews_corrcoef(train_targets, train_preds), "\t Validation MCC:", matthews_corrcoef(val_targets, val_preds))

########## define test data ##########

testset = []
for i in range(len(X_test)):
  testset.append([np.transpose(X_test[i]), y_test[i]])

# Data loader 
batchsize = 64
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=True)

########## test network ##########

test_probs, test_preds, test_targs = [], [], []
with torch.no_grad():
  for batch_idx, (data, target) in enumerate(testloader):
      x_batch_test = data.float().detach().cuda()
      y_batch_test = target.float().detach().unsqueeze(1).cuda()
      
      output = net(x_batch_test)

      test_batch_loss = criterion(output, y_batch_test)
      test_predicts = np.round(output.detach().cpu())
      test_targs += list(np.array(y_batch_test.cpu()))
      test_preds += list(test_predicts.numpy().flatten())

print("Test MCC:", matthews_corrcoef(test_targs, test_preds))

########## Performance ##########

def plot_roc(targets, predictions):
    # ROC
    fpr, tpr, threshold = metrics.roc_curve(targets, predictions)
    roc_auc = metrics.auc(fpr, tpr)

    # plot ROC
    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    #plt.show()


plot_roc(train_targets, train_preds)
plot_roc(val_targets, val_preds)
plot_roc(test_targs, test_preds)
