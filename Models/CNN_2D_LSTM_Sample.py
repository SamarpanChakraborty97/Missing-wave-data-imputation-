#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Module
from torch.nn import Conv1d
from torch.nn import Linear
from torch.nn import MaxPool1d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
import matplotlib.pyplot as plt
import math

data = pd.read_csv("Z_displacement.csv")
x = data.index
y = data['Eta']


# In[2]:


data1 = y[:577]

data2 = y[580:2950]

data3 = y[2954:4077]
data4 = y[4080:4731]
data5 = y[4734:5865]
data6 = y[5869:7423]
data7 = y[14445:14910]
data8 = y[14913:19719]
data9 = y[19722:20935]
data10 = y[20942:26367]
data11 = y[26933:29084]
data12 = y[29087:32218]
data13 = y[32222:39578]
data14 = y[39581:45077]
data15 = y[45080:46170]
data16 = y[46173:57830]
data17 = y[57835:]


# In[2]:


Full_TimeSeries = [y[:577],y[580:2950],y[2954:4077],y[4080:4731],y[4734:5865],y[5869:7423],y[7426:14442],y[14445:14910],y[14913:19719],y[19722:20935],y[20942:26367]]


# In[3]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[4]:


Eta_Df = pd.DataFrame([])
for num in range(len(Full_TimeSeries)):
    series = Full_TimeSeries[num]
    L = len(series)
    missing_len = int(0.1*L)

    data = np.asarray(series)
    data = scaler.fit_transform(data.reshape(-1,1))
    data[int(0.5*L)-int(missing_len/2):int(0.5*L)+int(missing_len/2)].fill(np.NaN)
    df = pd.DataFrame(data, columns = ['Eta'])
    Eta_Df = Eta_Df.append(df, ignore_index = True)


# In[5]:


def generateLaggedDf(df, training_len, missing_len, col_name):
    df_new = df.copy()
    for i in range(1,training_len + missing_len):
        df_new[f"Lag{i}"] = df[[col_name]].shift(i)
    df_new = df_new.iloc[training_len + missing_len:]
    
    df_new = df_new.dropna(axis= 0)
    
    train_len = int(0.7*len(df_new))
    
    df_train = df_new[:train_len]    
    df_val = df_new[train_len:]
    
    trainY = df_train.iloc[:,:missing_len]
    trainX = df_train.drop(df_train.iloc[:,:missing_len], axis=1)
    
    valY = df_val.iloc[:,:missing_len]
    valX = df_val.drop(df_train.iloc[:,:missing_len], axis=1)
    
    return trainX, trainY, valX, valY


# In[6]:


input_len = 200
output_len = 1
Train_X, Train_Y, Val_X, Val_Y = generateLaggedDf(Eta_Df, input_len, output_len,'Eta')


# In[7]:


X_Train = np.asarray(Train_X.iloc[:,::-1])
Y_Train = np.asarray(Train_Y.iloc[:,::-1])
X_Val = np.asarray(Val_X.iloc[:,::-1])
Y_Val = np.asarray(Val_Y.iloc[:,::-1])


# In[8]:


X_t = torch.Tensor(X_Train.copy())
X_v = torch.Tensor(X_Val.copy())


# In[9]:


len1 = len(np.arange(0,math.floor(0.7 * len(X_Train)),1))
len2 = len(np.arange(0,math.floor(0.7 * len(X_Val)),1))


# In[10]:


X_Train_LSTM_Tensor = torch.zeros(len1, input_len)
X_Val_LSTM_Tensor = torch.zeros(len2, input_len)


# In[11]:


for i in range(0,len1):
    X_Train_LSTM_Tensor[i] = X_t[i]

for j in range(0,len2):
    X_Val_LSTM_Tensor[j] = X_v[j]


# In[12]:


Y_t = torch.Tensor(Y_Train.copy())
Y_v = torch.Tensor(Y_Val.copy())

Y_Train_Tensor = torch.zeros(len1,output_len)
Y_Val_Tensor = torch.zeros(len2,output_len)

for i in range(0,len1):
    Y_Train_Tensor[i] = Y_t[i]

for j in range(0,len2):
    Y_Val_Tensor[j] = Y_v[j]


# In[13]:


import torchvision
import torchvision.transforms as transforms
from PIL import Image
transform = transforms.ToTensor()


# In[14]:


import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import datetime as datetime
from timeit import default_timer as timer


# In[15]:


X_Train_CNN_Tensor = torch.zeros((len1, 3 , 77, 77))
for k in range(0,len1) :
    image = Image.open(f'Sample_{k}.jpeg')
    tensor = transform(image)
    X_Train_CNN_Tensor[k] = tensor

X_Val_CNN_Tensor = torch.zeros((len2, 3 , 77, 77))
for k in range(0,len2) :
    image = Image.open(f'SampleVal_{k}.jpeg')
    tensor = transform(image)
    X_Val_CNN_Tensor[k] = tensor


# In[16]:


from torch.utils.data import Dataset

class CNN_LSTM_Dataset(Dataset):

    def __init__(self, images, entries, labels):
        self.X1 = images
        self.X2 = entries
        self.Y = labels

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, idx):
        data1 = self.X1[idx,:]
        data2 = self.X2[idx,:]

        data_labels = self.Y[idx,:]

        return (data1, data2, data_labels)


# In[17]:


trainZ = CNN_LSTM_Dataset(X_Train_CNN_Tensor, X_Train_LSTM_Tensor, Y_Train_Tensor)
valZ = CNN_LSTM_Dataset(X_Val_CNN_Tensor, X_Val_LSTM_Tensor, Y_Val_Tensor)


# In[18]:


import torch.nn as nn


# In[19]:


class CNN_LSTM_Module(nn.Module):
    def __init__(self, oc1, s_conv, ks, ts, i_dim, h_dim, l_dim, d_prob, mlp_hdim1, mlp_odim, a_net_feature):
        self.oc1 = oc1
        self.oc2 = math.floor(0.7 * self.oc1)
        self.oc3 = math.floor(0.7 * self.oc2)
        self.s_conv = s_conv
        self.ks = ks
        self.ts = ts
        self.i_dim = i_dim
        self.h_dim = h_dim
        self.l_dim = l_dim
        self.d_prob = d_prob
        self.mlp_hdim1 = mlp_hdim1
        self.mlp_odim = mlp_odim
        self.a_net_feature = a_net_feature

        super(CNN_LSTM_Module, self).__init__()

        "Convolution part of the model"

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=3, out_channels=oc1, kernel_size=ks)
        size1 = math.floor((ts[0] - ks)/s_conv)+1
        size2 = math.floor((ts[1] - ks)/s_conv)+1

        self.relu = ReLU()

        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=oc1, out_channels=self.oc2,kernel_size=ks)
        size1 = math.floor((size1 - ks)/s_conv)+1
        size2 = math.floor((size2 - ks)/s_conv)+1

         # initialize third set of CONV => RELU => POOL layers
        self.conv3 = Conv2d(in_channels=self.oc2, out_channels=self.oc3,kernel_size=ks)
        size1 = math.floor((size1 - ks)/s_conv)+1
        size2 = math.floor((size2 - ks)/s_conv)+1

        """LSTM part of the model"""
        # LSTM layers
        self.lstm = nn.LSTM(i_dim, h_dim, l_dim, batch_first=True, dropout=d_prob)

        """Attention module of the network"""
        in_features_Layer1 = (size1 * size2) + (h_dim * l_dim)
        out_features_Layer1 = a_net_feature
        self.attention1 = Linear(in_features_Layer1, out_features_Layer1)

        self.tanh = nn.Tanh()

        in_features_Layer2 = self.a_net_feature
        out_features_Layer2 = 1
        self.attention2 = Linear(in_features_Layer2, out_features_Layer2)

        self.smax = nn.Softmax(dim=1)

        """Fusion and Predictions using Multi Layer Perceptron"""
        fusion_input_dim = (size1 * size2) + (self.h_dim * self.l_dim)
        fusion_hidden_dim1 = math.floor(fusion_input_dim/2)
        fusion_output_dim = self.mlp_odim

        self.fc1 = nn.Linear(fusion_input_dim, fusion_hidden_dim1)
        self.sigmoid = nn.Sigmoid()
        self.relu = ReLU()
        self.fc2 = nn.Linear(fusion_hidden_dim1, fusion_output_dim)

    def forward(self, x1, x2):
        '''CNN 2D convolution using scaleograms'''
        x1 = self.conv1(x1)
        x1 = self.relu(x1)

        x1 = self.conv2(x1)
        x1 = self.relu(x1)

        x1 = self.conv3(x1)
        x1 = self.relu(x1)

        #print("The size of input after going through the convolutional layer has the shape of {}".format(x1.size()))

        x1_inter = x1.reshape(x2.size(0), -1, x1.shape[2]*x1.shape[3])
        #print("The size of the reshaped convolutional layer has the shape of {}".format(x1_inter.size()))

        '''Hidden state prediction using LSTM for the global trend'''
        h0 = torch.zeros(self.l_dim, x2.size(0), self.h_dim).requires_grad_().to(device)

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.l_dim, x2.size(0), self.h_dim).requires_grad_().to(device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x2, (h0.detach(), c0.detach()))

        hn = hn.reshape([x2.size(0),-1])
        hn_rec= hn
        for l in range(1, x1_inter.shape[1]):
            hn_rec = torch.cat((hn_rec, hn),1)

        #print(hn_rec.shape)

        hn_rec = hn_rec.reshape([x2.size(0),x1_inter.size(1),hn.size(1)])
        #print("The size of the reconstructed LSTM hidden layer has the shape of {}".format(hn_rec.size()))
        '''Attention module using the hidden states and the CNN module'''
        x = torch.concat((x1_inter,hn_rec),dim=2)
        #print("The size of the augmented input attention layer has the shape of {}".format(x.size()))
        x = self.attention1(x)
        #print("The size of the input after the first attention layer has the shape of {}".format(x.size()))
        x = self.tanh(x)
        x = self.attention2(x)
        #print("The size of the input after the second attention layer has the shape of {}".format(x.size()))
        x = self.smax(x)
        #print("The size of the input after the softmax layer has the shape of {}".format(x.size()))
        #print(torch.transpose(x1_inter,1,2).shape)
        x = torch.bmm(torch.transpose(x1_inter,1,2),x).to(device)
        #print("The size of the input after the weighted sum has the shape of {}".format(x.size()))
        x = x.reshape([x2.size(0),-1])
        #print("The size of the reshaped input layer before MLP has the shape of {}".format(x.size()))

        '''Fusion and prediction using MLP'''
        x_MLP = torch.concat((x,hn),dim=1)
        #print("The size of the augmented input MLP layer` has the shape of {}".format(x_MLP.size()))
        x_MLP = self.fc1(x_MLP)
        #print("The size of the input after the first hidden MLP layer has the shape of {}".format(x_MLP.size()))
        x_MLP = self.relu(x_MLP)
        x_MLP = self.fc2(x_MLP)
        output = self.relu(x_MLP)
        #print("The size of the output after the MLP layer has the shape of {}".format(output.size()))
        
        return output


# In[20]:


class Optimization:
    """Optimization is a helper class that allows training, validation, prediction.
    """
    def __init__(self, model, loss_fn, optimizer, patience, min_delta = 1e-5):

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.counter = 0
        self.min_delta = min_delta
        self.min_validation_loss = np.inf
        self.patience = patience
        
    def train_step(self,x1,x2,y):
        self.model.train()

        yhat = self.model(x1,x2)
        loss = self.loss_fn(y, yhat)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()
    
    def earlyStop(self, validation_loss):
        if validation_loss < (self.min_validation_loss - self.min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0
            
        elif validation_loss >= (self.min_validation_loss - self.min_delta):
            self.counter +=1
            if self.counter >= self.patience:
                return True
            return False

    def train(self, train_loader, val_loader, batch_size, n_epochs, n_features):

        model_path = f'cnn_lstm_with_training_length_{n_features}.pt'
        break_out_flag = False

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x1_batch, x2_batch, y_batch in train_loader:
                x2_batch = x2_batch.view([batch_size, -1, n_features]).to(device)
                x1_batch = x1_batch.to(device)
                y_batch = y_batch.to(device)
                yhat = self.model(x1_batch, x2_batch)
                loss = self.train_step(x1_batch, x2_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x1_val, x2_val, y_val in val_loader:
                    x2_val = x2_val.view([batch_size, -1, n_features]).to(device)
                    x1_val = x1_val.to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x1_val, x2_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)
                if self.earlyStop(validation_loss):
                    break_out_flag = True
                    break               
            
            if break_out_flag:
                torch.save(self.model.state_dict(), model_path)
                break

            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )
        torch.save(self.model.state_dict(), model_path)

        
    def evaluate(self, x, test):
        with torch.no_grad():
            predictions = []
            values = []
            for i in range(len(test)):
                x = x.to(device)
                self.model.eval()
                x_test = x.view([1, -1, 100]).to(device)
                yhat = self.model(x_test)
                predictions.append(yhat.to(device).detach().numpy())
                values.append(test[i].to(device).detach().numpy())
                #print(x.size())
                #print(yhat.size())
                x=torch.reshape(x,(100,1))
                x = torch.cat((x,yhat),0)
                x = x[1:]

        return predictions, values


    def plot_losses(self, training_len):
        """The method plots the calculated loss values for training and validation
        """
        np.savetxt(f"CNN_LSTM_Training_length={training_len}_train.out", self.train_losses, fmt='%1.4e')
        np.savetxt(f"CNN_LSTM_Training_length={training_len}_val.out", self.val_losses, fmt='%1.4e')
        
        plt.figure(figsize=[10,8])
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title(f"Losses for training length = {training_len}")
        plt.grid()
        plt.show()
        plt.savefig(f'CNN_LSTM_Losses comparisons for training length={training_len} over epochs.png',dpi=300)
        plt.close()


# In[21]:


# import the necessary packages
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import LogSoftmax


# In[22]:


start = timer()

input_dim = input_len
output_dim = output_len

oc1 = 200
ks = 3
hidden_dim = 32
layer_dim = 2
dropout_prob = 0.1
a_net_feature = 60
mlp_hiddendim1 = 150
mlp_odim = output_len

oc1 = 100
tuple_shape = (77, 77)
weight_decay = 1e-6
ks = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_LSTM_Module(oc1 = oc1, s_conv = 1, ks = ks, ts = tuple_shape, i_dim = input_dim, h_dim = hidden_dim, l_dim = layer_dim, d_prob = dropout_prob, mlp_hdim1 = mlp_hiddendim1, mlp_odim = output_len, a_net_feature = a_net_feature)
model = model.to(device)

loss_fn = nn.MSELoss(reduction="mean")

learning_rate = 0.00001
batch_size = 128
n_epochs = 1000

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
Ftrain_loader = DataLoader(trainZ, batch_size=batch_size, shuffle=False, drop_last=True)
Fval_loader = DataLoader(valZ, batch_size=batch_size, shuffle=False, drop_last=True)

opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer, patience = 10)
opt.train(Ftrain_loader, Fval_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
opt.plot_losses(input_dim)
            
end = timer()

dur = (end-start)/60
print(f'The total duration for the training is {dur} minutes')


# In[ ]:




