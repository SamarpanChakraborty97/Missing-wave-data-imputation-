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

import matplotlib.pyplot as plt
import math
import torchvision
import torchvision.transforms as transforms
from PIL import Image
transform = transforms.ToTensor()
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import datetime as datetime
from timeit import default_timer as timer
import pywt


# In[28]:


x = np.arange(0,10*math.pi, math.pi/600)


# In[29]:


high_freq = 0.1 * np.sin(20*x)


# In[30]:


low_freq = np.sin(2*x)


# In[31]:


plt.figure(figsize=[15,5])
#plt.plot(x, low_freq, 'b', linewidth = 0.6)
#plt.plot(x, high_freq, 'r', linewidth = 0.6)
plt.plot(x, high_freq+low_freq, 'k', linewidth = 0.6)


# In[32]:


data = np.asarray(high_freq + low_freq)


# In[33]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data = scaler.fit_transform(data.reshape(-1,1))


# In[34]:


test_len = int(0.05 * len(data))


# In[38]:


test_data = data[-test_len:]
train_val_data = data[:-test_len]


# In[39]:


print(len(train_val_data))
print(len(test_data))

train_len = int(0.75 * len(train_val_data))

train_data = train_val_data[:train_len]
val_data = train_val_data[train_len:len(train_val_data)]
# In[44]:


Eta_Df = pd.DataFrame([])

df = pd.DataFrame(train_val_data, columns = ['Eta'])
Eta_Df = Eta_Df.append(df, ignore_index = True)


# In[46]:


def generateLaggedDf(df, training_len, missing_len, col_name):
    df_new = df.copy()
    for i in range(1,training_len + missing_len):
        df_new[f"Lag{i}"] = df[[col_name]].shift(i)
    df_new = df_new.iloc[training_len + missing_len:]
    
    df_new = df_new.dropna(axis= 0)
    L = len(df_new)

    df_train = df_new[:int(0.8*L)]
    df_val = df_new[int(0.8*L):L]
    
    trainY = df_train.iloc[:,:missing_len]
    trainX = df_train.drop(df_train.iloc[:,:missing_len], axis=1)
    
    valY = df_val.iloc[:,:missing_len]
    valX = df_val.drop(df_train.iloc[:,:missing_len], axis=1)
    
    return trainX, trainY, valX, valY


# In[47]:


input_len = 500
output_len = 1
Train_X, Train_Y, Val_X, Val_Y = generateLaggedDf(Eta_Df, input_len, output_len,'Eta')


# In[50]:


X_Train = np.asarray(Train_X.iloc[:,::-1])
Y_Train = np.asarray(Train_Y.iloc[:,::-1])
X_Val = np.asarray(Val_X.iloc[:,::-1])
Y_Val = np.asarray(Val_Y.iloc[:,::-1])

torch.manual_seed(2)


X_t = torch.Tensor(X_Train.copy())
X_v = torch.Tensor(X_Val.copy())


len1 = len(np.arange(0,math.floor(0.7 * len(X_Train)),1))
len2 = len(np.arange(0,math.floor(0.7 * len(X_Val)),1))


Y_t = torch.Tensor(Y_Train.copy())
Y_v = torch.Tensor(Y_Val.copy())

Y_Train_Tensor = torch.zeros(len(Y_Train),output_len)
Y_Val_Tensor = torch.zeros(len(Y_Val),output_len)


for i in range(len(Y_Train_Tensor)):    
    Y_Train_Tensor[i] = Y_t[i]

for j in range(len(Y_Val_Tensor)):
    Y_Val_Tensor[j] = Y_v[j]
    

X_Train_CNN_Tensor = torch.zeros(len(X_Train), 3 , 77, 77)
for k in range(len(X_Train_CNN_Tensor)):
    image = Image.open(f'Sample_{k}.jpeg')
    tensor = transform(image)
    X_Train_CNN_Tensor[k] = tensor

X_Val_CNN_Tensor = torch.zeros(len(X_Val), 3 , 77, 77)
for k in range(len(X_Val_CNN_Tensor)) :
    image = Image.open(f'SampleVal_{k}.jpeg')
    tensor = transform(image)
    X_Val_CNN_Tensor[k] = tensor


from torch.utils.data import Dataset

class CNN_Dataset(Dataset):

    def __init__(self, images, labels):
        self.X1 = images
        self.Y = labels

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, idx):
        data1 = self.X1[idx,:]
        data_labels = self.Y[idx,:]

        return (data1, data_labels)


trainZ = CNN_Dataset(X_Train_CNN_Tensor, Y_Train_Tensor)
valZ = CNN_Dataset(X_Val_CNN_Tensor, Y_Val_Tensor)


import torch.nn as nn


class CNN_Module(nn.Module):
    '''
    Models a simple convolutional neural network
    '''
    def __init__(self, out_features, ts, s_conv, oc1, ks, dil1, dil2, i_dim, d_prob):
        
        self.oc1 = oc1       
        self.s_conv = s_conv
        self.ks = ks
        self.ts = ts
        self.dil1 = dil1
        self.dil2 = dil2
        self.i_dim = i_dim
        self.out_features = out_features
        self.d_prob = d_prob

        # call the parent constructor
        super(CNN_Module, self).__init__()
        
        self.convD1_1 = Conv2d(in_channels=3, out_channels=oc1, dilation = dil1, kernel_size=ks)
        self.convD1_2 = Conv2d(in_channels=oc1, out_channels=oc1, dilation = dil1, kernel_size=ks)
        
        #self.convD2_1 = Conv2d(in_channels=3, out_channels=oc1, dilation = dil2, kernel_size=ks)
        #self.convD2_2 = Conv2d(in_channels=oc1, out_channels=oc1, dilation = dil2, kernel_size=ks)
        
        self.relu = ReLU()
        self.dropout = nn.Dropout(self.d_prob)
        
        ### DILATION VAL 1 ###
        
        ### initialize first set of CONV => RELU => layers ###
        size1_1 = math.floor((ts[0] - dil1 * (ks-1) - 1)/s_conv)+1
        size1_2 = math.floor((ts[1] - dil1 * (ks-1) - 1)/s_conv)+1
        
        ### initialize second set of CONV => RELU => layers ###
        size1_1 = math.floor((size1_1 - dil1 * (ks-1) - 1)/s_conv)+1
        size1_2 = math.floor((size1_2 - dil1 * (ks-1) - 1)/s_conv)+1
        
        ### initialize third set of CONV => RELU => layers ###
        size1_1 = math.floor((size1_1 - dil1 * (ks-1) - 1)/s_conv)+1
        size1_2 = math.floor((size1_2 - dil1 * (ks-1) - 1)/s_conv)+1
             
        
        ### DILATION VAL 2 ###
        
        ### initialize first set of CONV => RELU => layers ###
        #size2_1 = math.floor((ts[0] - dil2 * (ks-1) - 1)/s_conv)+1
        #size2_2 = math.floor((ts[1] - dil2 * (ks-1) - 1)/s_conv)+1
        
        ### initialize second set of CONV => RELU => layers ###
        #size2_1 = math.floor((size2_1 - dil2 * (ks-1) - 1)/s_conv)+1
        #size2_2 = math.floor((size2_2 - dil2 * (ks-1) - 1)/s_conv)+1
        
        ### initialize third set of CONV => RELU => layers ###
        #size2_1 = math.floor((size2_1 - dil2 * (ks-1) - 1)/s_conv)+1
        #size2_2 = math.floor((size2_2 - dil2 * (ks-1) - 1)/s_conv)+1
        
        #print(size)
        self.fc1 = Linear(in_features= self.oc1 * (size1_1 * size1_2), out_features=i_dim)
        self.fc2 = Linear(in_features=i_dim, out_features=out_features)
        
    def forward(self, x1):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        
        #### 1st dilation variant ####
        
        xD1 = self.convD1_1(x1)
        xD1 = self.relu(xD1)

        xD1 = self.convD1_2(xD1)
        xD1 = self.relu(xD1)

        xD1 = self.convD1_2(xD1)
        xD1 = self.relu(xD1)
        
        #### 2nd dilation variant ####
        
        #xD2 = self.convD2_1(x1)
        #xD2 = self.relu(xD2)

        #xD2 = self.convD2_2(xD2)
        #xD2 = self.relu(xD2)

        #xD2 = self.convD2_2(xD2)
        #xD2 = self.relu(xD2)

        xd1 = flatten(xD1, 1)
        #xd2 = flatten(xD2, 1)
        #x = torch.cat((xd1, xd2), 1)
        
        x = self.fc1(xd1)
        
        x = self.relu(x)
        
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        output = x
        
        return output


# In[18]:


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
        
    def train_step(self, x, y):

        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
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

    def train(self, train_loader, val_loader, batch_size, n_epochs, n_features, output_len):

        model_path = f'cnn_2d_twoFrequencies.pt'
        break_out_flag = False

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                yhat = self.model(x_batch)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
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


    def plot_losses(self, output_len):
        """The method plots the calculated loss values for training and validation
        """
        #np.savetxt(f"CNN_2D_Output_length={output_len}_train.out", self.train_losses, fmt='%1.4e')
        #np.savetxt(f"CNN_2D_Output_length={output_len}_val.out", self.val_losses, fmt='%1.4e')
        
        #plt.figure(figsize=[10,8])
        #plt.plot(self.train_losses, label="Training loss")
        #plt.plot(self.val_losses, label="Validation loss")
        #plt.legend()
        #plt.title(f"Losses for output length = {output_len}")
        #plt.grid()
        #plt.show()
        #plt.savefig(f'CNN 2D Losses comparisons for output length={output_len} over epochs.png',dpi=300)
        #plt.close()


# In[19]:


# import the necessary packages
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import LogSoftmax


# In[21]:


start = timer()

input_dim = input_len
output_dim = output_len
dropout = 0.3

i_dim = 150
oc1 = 150

dilation1 = 1
dilation2 = 7

tuple_shape = (77, 77)
weight_decay = 1e-3
ks = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_Module(out_features = output_dim, ts = tuple_shape, s_conv = 1,  oc1 = oc1, ks = ks, dil1 = dilation1, dil2 = dilation2, i_dim = i_dim, d_prob = dropout)
model = model.to(device)

loss_fn = nn.MSELoss(reduction="mean")

learning_rate = 1e-5
batch_size = 32
n_epochs = 1000

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
Ftrain_loader = DataLoader(trainZ, batch_size=batch_size, shuffle=False, drop_last=True)
Fval_loader = DataLoader(valZ, batch_size=batch_size, shuffle=False, drop_last=True)

opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer, patience = 30)
opt.train(Ftrain_loader, Fval_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim, output_len = output_len)
opt.plot_losses(output_dim)
            
end = timer()

dur = (end-start)/60
print(f'The total duration for the training is {dur} minutes')