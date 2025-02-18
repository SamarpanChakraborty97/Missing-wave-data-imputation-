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


# In[51]:


import torch
import torch.nn as nn


# In[52]:


import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import datetime as datetime
from timeit import default_timer as timer
import pywt


# In[53]:


X_t = torch.Tensor(X_Train.copy())
X_v = torch.Tensor(X_Val.copy())


# In[54]:


import torchvision
import torchvision.transforms as transforms
from PIL import Image
transform = transforms.ToTensor()


# In[55]:


import math


# In[56]:


#len1 = len(np.arange(0,math.floor(0.7 * len(X_Train)),1))
#len2 = len(np.arange(0,math.floor(0.7 * len(X_Val)),1))


# In[57]:


X_Train_LSTM_Tensor = torch.zeros(len(X_Train), input_len)
X_Val_LSTM_Tensor = torch.zeros(len(X_Val), input_len)

#X_Train_LSTM_Tensor = torch.zeros(20, input_len)
#X_Val_LSTM_Tensor = torch.zeros(20, input_len)


# In[ ]:


for i in range(len(X_Train_LSTM_Tensor)):
    X_Train_LSTM_Tensor[i] = X_t[i]

for j in range(len(X_Val_LSTM_Tensor)):
    X_Val_LSTM_Tensor[j] = X_v[j]


# In[ ]:


Y_t = torch.Tensor(Y_Train.copy())
Y_v = torch.Tensor(Y_Val.copy())

Y_Train_Tensor = torch.zeros(len(Y_Train),output_len)
Y_Val_Tensor = torch.zeros(len(Y_Val),output_len)

#Y_Train_Tensor = torch.zeros(20,output_len)
#Y_Val_Tensor = torch.zeros(20,output_len)

for i in range(len(Y_Train_Tensor)):    
    Y_Train_Tensor[i] = Y_t[i]

for j in range(len(Y_Val_Tensor)):
    Y_Val_Tensor[j] = Y_v[j]


# In[ ]:


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


X_Train_CNN_Tensor.shape


X_Train_CNN_Tensor


# In[ ]:


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


# In[ ]:


trainZ = CNN_LSTM_Dataset(X_Train_CNN_Tensor, X_Train_LSTM_Tensor, Y_Train_Tensor)
valZ = CNN_LSTM_Dataset(X_Val_CNN_Tensor, X_Val_LSTM_Tensor, Y_Val_Tensor)


# In[ ]:


import torch.nn as nn


# In[ ]:


class CNN_LSTM_Module(nn.Module):
    def __init__(self, oc1, s_conv, ks, dil1, dil2, ts, i_dim, h_dim, l_dim, d_prob, mlp_hdim1, mlp_odim, a_net_feature):
        self.oc1 = oc1
        self.s_conv = s_conv
        self.ks = ks
        self.ts = ts
        self.dil1 = dil1
        self.dil2 = dil2
        
        self.i_dim = i_dim
        self.h_dim = h_dim
        self.l_dim = l_dim
        self.d_prob = d_prob
        self.mlp_hdim1 = mlp_hdim1
        self.mlp_odim = mlp_odim
        self.a_net_feature = a_net_feature

        super(CNN_LSTM_Module, self).__init__()

        '''Convolution part of the model'''
        
        self.convD1_1 = Conv2d(in_channels=3, out_channels=oc1, dilation = dil1, kernel_size=ks)
        self.convD1_2 = Conv2d(in_channels=oc1, out_channels=oc1, dilation = dil1, kernel_size=ks)
        
        #self.convD2_1 = Conv2d(in_channels=3, out_channels=oc1, dilation = dil2, kernel_size=ks)
        #self.convD2_2 = Conv2d(in_channels=oc1, out_channels=oc1, dilation = dil2, kernel_size=ks)
        
        self.relu = ReLU()
        
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
        

        """LSTM part of the model"""
        # LSTM layers
        self.lstm = nn.LSTM(i_dim, h_dim, l_dim, batch_first=True, dropout=0.4)

        """Attention module of the network"""
        out_features_Layer1 = a_net_feature
        
        D1_in_features_Layer1 = (size1_1 * size1_2) + (h_dim * l_dim)
        self.attentionD1 = Linear(D1_in_features_Layer1, out_features_Layer1)
        
        #D2_in_features_Layer1 = (size2_1 * size2_2) + (h_dim * l_dim)
        #self.attentionD2 = Linear(D2_in_features_Layer1, out_features_Layer1)

        self.tanh = nn.Tanh()

        in_features_Layer2 = self.a_net_feature
        out_features_Layer2 = 1
        self.attention2 = Linear(in_features_Layer2, out_features_Layer2)

        self.smax = nn.Softmax(dim=1)

        """Fusion and Predictions using Multi Layer Perceptron"""
        
        fusion_input_dim = (size1_1 * size1_2) + (self.h_dim * self.l_dim)
        fusion_hidden_dim1 = math.floor(fusion_input_dim/2)
        fusion_output_dim = self.mlp_odim

        self.fc1 = nn.Linear(fusion_input_dim, fusion_hidden_dim1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(self.d_prob)
        self.relu = ReLU()
        self.fc2 = nn.Linear(fusion_hidden_dim1, fusion_output_dim)

    def forward(self, x1, x2):
        '''CNN 2D convolution using scaleograms'''
        
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
        

        #print("The size of input after going through the convolutional layer has the shape of {}".format(x1.size()))

        xD1_inter = xD1.reshape(x2.size(0), -1, xD1.shape[2]*xD1.shape[3])
        #xD2_inter = xD2.reshape(x2.size(0), -1, xD2.shape[2]*xD2.shape[3])
        
        print("The size of the reshaped convolutional layer has the shape of {}".format(xD1_inter.size()))

        '''Hidden state prediction using LSTM for the global trend'''
        h0 = torch.zeros(self.l_dim, x2.size(0), self.h_dim).requires_grad_().to(device)

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.l_dim, x2.size(0), self.h_dim).requires_grad_().to(device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x2, (h0.detach(), c0.detach()))
        
        #print("The size of the hidden LSTM output directly has the shape of {}".format(hn.size()))
        
        #a = torch.Tensor([[]]).to(device)
        #for i in range(hn.shape[0]):
        #    a = torch.cat((a,hn[i]),1)
        
        hn = torch.cat((hn[0], hn[1]), 1)
        
        #hn = hn.reshape([x2.size(0),-1])
        #print("The size of the hidden LSTM output has the shape of {}".format(hn.size()))
        
        hn_rec1 = hn.reshape(hn.shape[0],1,hn.shape[1])
        for l in range(1, xD1_inter.shape[1]):
            hn_rec1 = torch.cat((hn_rec1, hn.reshape(hn.shape[0],1,hn.shape[1])),1)
        
        #hn_rec2 = hn.reshape(hn.shape[0],1,hn.shape[1])
        #for l in range(1, xD2_inter.shape[1]):
        #    hn_rec2 = torch.cat((hn_rec2, hn.reshape(hn.shape[0],1,hn.shape[1])),1)

        #print("The size of the hidden LSTM output after concatenation has the shape of {}".format(hn_rec.size()))

        #hn_rec = hn_rec.reshape([-1,x1_inter.size(1),hn.size(1)])
        #print("The size of the reconstructed LSTM hidden layer has the shape of {}".format(hn_rec.size()))
        
        '''Attention module using the hidden states and the CNN module'''
        #xd1 = torch.concat((xD1_inter,hn_rec1),dim=2)
        #xd2 = torch.concat((xD2_inter,hn_rec2),dim=2)
        
        #print("The size of the augmented input attention layer has the shape of {}".format(x.size()))
        
        #xdil1 = self.attentionD1(xd1)
        #xdil2 = self.attentionD2(xd2)
        
        #print("The size of the input after the first attention layer has the shape of {}".format(x.size()))
        
        #xdil1 = self.tanh(xdil1)
        #xdil2 = self.tanh(xdil2)
        
        #xdil1 = self.attention2(xdil1)
        #xdil2 = self.attention2(xdil2)
        
        #print("The size of the input after the second attention layer has the shape of {}".format(x.size()))
        
        #xdil1 = self.smax(xdil1)
        #xdil2 = self.smax(xdil2)
        
        #print("The size of the input after the softmax layer has the shape of {}".format(x.size()))
        
        #print(torch.transpose(x1_inter,1,2).shape)
        
        #xdil1 = torch.bmm(torch.transpose(xD1_inter,1,2),xdil1).to(device)
        #xdil2 = torch.bmm(torch.transpose(xD2_inter,1,2),xdil2).to(device)
        
        #print("The size of the input after the weighted sum has the shape of {}".format(x.size()))
        
        #xdil1 = xdil1.reshape([x2.size(0),-1])
        #xdil2 = xdil2.reshape([x2.size(0),-1])
        
        #print("The size of the reshaped input layer before MLP has the shape of {}".format(x.size()))

        '''Fusion and prediction using MLP'''
        #x_MLP = torch.concat((xdil1, xdil2, hn),dim=1)
        #x_MLP = torch.concat((xdil1, hn),dim=1)
        
        #x_d1 = xD1_inter.reshape([x2.size(0),-1])
        x_d1 = torch.amax(xD1_inter, 1)
        
        #print("The size of the maxed CNN output layer has the shape of {}".format(x_d1.size()))
        
        x_MLP = torch.concat((x_d1,hn),dim=1)
               
        #print("The size of the augmented input MLP layer has the shape of {}".format(x_MLP.size()))
        
        x_MLP = self.fc1(x_MLP)
        
        #print("The size of the input after the first hidden MLP layer has the shape of {}".format(x_MLP.size()))
        
        x_MLP = self.relu(x_MLP)
        
        x_MLP = self.dropout(x_MLP)
        
        x_MLP = self.fc2(x_MLP)
        
        output = x_MLP
        
        #print("The size of the output after the MLP layer has the shape of {}".format(output.size()))
        
        return output


# In[ ]:


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

    def train(self, train_loader, val_loader, batch_size, n_epochs, n_features, output_dim):

        model_path = f'cnn_lstm_withoutAttention_twoFreqs.pt'
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

            if (epoch <= 10) | (epoch % 25 == 0):
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
        #np.savetxt(f"CNN_LSTM_dilatedConcatenated_79_mean_train.out", self.train_losses, fmt='%1.4e')
        #np.savetxt(f"CNN_LSTM_dilatedConcatenated_79_mean_val.out", self.val_losses, fmt='%1.4e')
        
        #plt.figure(figsize=[10,8])
        #plt.plot(self.train_losses, label="Training loss")
        #plt.plot(self.val_losses, label="Validation loss")
        #plt.legend()
        #plt.title(f"Losses for dilated concatenation using mean reduction for 79 steps")
        #plt.grid()
        #plt.show()
        #plt.savefig(f'CNN_LSTM Losses comparisons for dilatedConcatenated_79_mean over epochs.png',dpi=300)
        #plt.close()


# In[ ]:


# import the necessary packages
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import LogSoftmax


start = timer()

input_dim = input_len
output_dim = output_len

ks = 3
hidden_dim = 128
layer_dim = 2
dropout_prob = 0.3
a_net_feature = 64
mlp_hiddendim1 = 150
mlp_odim = output_len

oc1 = 150

dilation1 = 1
dilation2 = 7

tuple_shape = (77, 77)
weight_decay = 1e-3
ks = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_LSTM_Module(oc1 = oc1, s_conv = 1, ks = ks, dil1 = dilation1, dil2 = dilation2, ts = tuple_shape, i_dim = input_dim, h_dim = hidden_dim, l_dim = layer_dim, d_prob = dropout_prob, mlp_hdim1 = mlp_hiddendim1, mlp_odim = output_len, a_net_feature = a_net_feature)
model = model.to(device)

loss_fn = nn.MSELoss(reduction="mean")

learning_rate = 1e-5
batch_size = 32
n_epochs = 1500

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
Ftrain_loader = DataLoader(trainZ, batch_size=batch_size, shuffle=False, drop_last=True)
Fval_loader = DataLoader(valZ, batch_size=batch_size, shuffle=False, drop_last=True)

opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer, patience = 30)
opt.train(Ftrain_loader, Fval_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim, output_dim = output_len)
opt.plot_losses(output_dim)
            
end = timer()

dur = (end-start)/60
print(f'The total duration for the training is {dur} minutes')