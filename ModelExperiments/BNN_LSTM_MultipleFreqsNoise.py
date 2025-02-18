#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pandas as pd
import numpy as np
import torch.nn as nn
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
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[12]:


import torch
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import datetime as datetime
from timeit import default_timer as timer
import pywt


# In[15]:


import torchvision
import torchvision.transforms as transforms
from PIL import Image
transform = transforms.ToTensor()
import math


# In[ ]:


x = np.arange(0,12*math.pi, math.pi/600)


first_freq = 0.4 * np.sin(20*x)
second_freq = 0.4 * np.sin(10*x)
third_freq = 0.4 * np.sin(3*x)
fourth_freq = 0.4 * np.sin(x)
fifth_freq = 0.4 * np.sin(x/2)
sixth_freq = 0.4 * np.sin(x/5)


data = np.asarray(first_freq + second_freq + third_freq + fourth_freq + fifth_freq + sixth_freq)
noise = np.random.normal(0,0.1,data.shape)
data = data + noise


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data = scaler.fit_transform(data.reshape(-1,1))

test_len = int(0.1 * len(data))

test_data = data[-test_len:]
train_val_data = data[:-test_len]

print(len(train_val_data))
print(len(test_data))

train_len = int(0.75 * len(train_val_data))

train_data = train_val_data[:train_len]
val_data = train_val_data[train_len:len(train_val_data)]

Eta_Df = pd.DataFrame([])

df = pd.DataFrame(train_val_data, columns = ['Eta'])
Eta_Df = Eta_Df.append(df, ignore_index = True)


# In[ ]:


input_len = 500
output_len = 1


# In[ ]:


from torch.utils.data import Dataset
class Bayesian_CNN_LSTM_Dataset(Dataset):

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


trainZ = Bayesian_CNN_LSTM_Dataset(X_Train_CNN_Tensor, X_Train_LSTM_Tensor, Y_Train_Tensor)
valZ = Bayesian_CNN_LSTM_Dataset(X_Val_CNN_Tensor, X_Val_LSTM_Tensor, Y_Val_Tensor)


# In[ ]:


class BayesianNet(nn.Module):
    '''
    Models a Bayesian convolutional neural network
    '''
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

        super(BayesianNet, self).__init__()
        
        '''Convolution part of the model'''
        
        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = bnn.BayesConv2d(prior_mu=0, prior_sigma=0.01,in_channels=3, out_channels=oc1,kernel_size=ks)
        self.conv2 = bnn.BayesConv2d(prior_mu=0, prior_sigma=0.01,in_channels=oc1, out_channels=oc1,kernel_size=ks)
        
        size1 = math.floor((ts[0] - ks)/s_conv)+1
        size2 = math.floor((ts[1] - ks)/s_conv)+1

        self.relu = ReLU()

        ### initialize first set of CONV => RELU => layers ###
        size1_1 = math.floor((ts[0] - dil1 * (ks-1) - 1)/s_conv)+1
        size1_2 = math.floor((ts[1] - dil1 * (ks-1) - 1)/s_conv)+1
        
        ### initialize second set of CONV => RELU => layers ###
        size1_1 = math.floor((size1_1 - dil1 * (ks-1) - 1)/s_conv)+1
        size1_2 = math.floor((size1_2 - dil1 * (ks-1) - 1)/s_conv)+1
        
        ### initialize third set of CONV => RELU => layers ###
        size1_1 = math.floor((size1_1 - dil1 * (ks-1) - 1)/s_conv)+1
        size1_2 = math.floor((size1_2 - dil1 * (ks-1) - 1)/s_conv)+1
        
        """LSTM part of the model"""
        # LSTM layers
        self.lstm = nn.LSTM(i_dim, h_dim, l_dim, batch_first=True, dropout=0.4)

        """Attention module of the network"""
        out_features_Layer1 = a_net_feature
        
        D1_in_features_Layer1 = (size1_1 * size1_2) + (h_dim * l_dim)
        self.attentionD1 = Linear(D1_in_features_Layer1, out_features_Layer1)

        self.tanh = nn.Tanh()

        in_features_Layer2 = self.a_net_feature
        out_features_Layer2 = 1
        self.attention2 = Linear(in_features_Layer2, out_features_Layer2)

        self.smax = nn.Softmax(dim=1)
        
        #print(size)
        """Fusion and Predictions using Multi Layer Perceptron"""
        
        #fusion_input_dim = (size1_1 * size1_2) + (size2_1 * size2_2) + (self.h_dim * self.l_dim)
        fusion_input_dim = (size1_1 * size1_2) + (self.h_dim * self.l_dim)
        fusion_hidden_dim1 = math.floor(fusion_input_dim/2)
        fusion_output_dim = self.mlp_odim
        
        #self.fc1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.005,in_features= self.oc4 * size1 * size2, out_features=i_dim)
        self.fc1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.01,in_features= fusion_input_dim, out_features=fusion_hidden_dim1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(self.d_prob)
        self.relu = ReLU()
        self.fc2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.01,in_features=fusion_hidden_dim1, out_features=fusion_output_dim)

    def forward(self, x1, x2):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        
        #print(x.shape)
        xd1 = self.conv1(x1)
        xd1 = self.relu(xd1)
        
        xd1 = self.conv2(xd1)
        xd1 = self.relu(xd1)
        
        xd1 = self.conv2(xd1)
        xd1 = self.relu(xd1)
        
        xd1_inter = xd1.reshape(x2.size(0), -1, xd1.shape[2]*xd1.shape[3])
        
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
        for l in range(1, xd1_inter.shape[1]):
            hn_rec1 = torch.cat((hn_rec1, hn.reshape(hn.shape[0],1,hn.shape[1])),1)
        
        #hn_rec2 = hn.reshape(hn.shape[0],1,hn.shape[1])
        #for l in range(1, xD2_inter.shape[1]):
        #    hn_rec2 = torch.cat((hn_rec2, hn.reshape(hn.shape[0],1,hn.shape[1])),1)

        #print("The size of the hidden LSTM output after concatenation has the shape of {}".format(hn_rec.size()))

        #hn_rec = hn_rec.reshape([-1,x1_inter.size(1),hn.size(1)])
        #print("The size of the reconstructed LSTM hidden layer has the shape of {}".format(hn_rec.size()))
        
        '''Attention module using the hidden states and the CNN module'''
        xd1 = torch.concat((xd1_inter,hn_rec1),dim=2)
        #xd2 = torch.concat((xD2_inter,hn_rec2),dim=2)
        
        #print("The size of the augmented input attention layer has the shape of {}".format(x.size()))
        
        xdil1 = self.attentionD1(xd1)
        #xdil2 = self.attentionD2(xd2)
        
        #print("The size of the input after the first attention layer has the shape of {}".format(x.size()))
        
        xdil1 = self.tanh(xdil1)
        #xdil2 = self.tanh(xdil2)
        
        xdil1 = self.attention2(xdil1)
        #xdil2 = self.attention2(xdil2)
        
        #print("The size of the input after the second attention layer has the shape of {}".format(x.size()))
        
        xdil1 = self.smax(xdil1)
        #xdil2 = self.smax(xdil2)
        
        #print("The size of the input after the softmax layer has the shape of {}".format(x.size()))
        
        #print(torch.transpose(x1_inter,1,2).shape)
        
        xdil1 = torch.bmm(torch.transpose(xd1_inter,1,2),xdil1).to(device)
        #xdil2 = torch.bmm(torch.transpose(xD2_inter,1,2),xdil2).to(device)
        
        #print("The size of the input after the weighted sum has the shape of {}".format(x.size()))
        
        xdil1 = xdil1.reshape([x2.size(0),-1])
        #xdil2 = xdil2.reshape([x2.size(0),-1])
        
        '''Fusion and prediction using MLP'''
        #x_MLP = torch.concat((xdil1, xdil2, hn),dim=1)
        x_MLP = torch.concat((xdil1, hn),dim=1)
        
        #print("The size of the augmented input MLP layer` has the shape of {}".format(x_MLP.size()))
        
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

    def __init__(self, model, loss_fn_mse, loss_fn_KL, optimizer, patience, min_delta = 1e-5, kl_weight = 0.05):
        self.model = model
        self.loss_fn_mse = loss_fn_mse
        self.loss_fn_KL = loss_fn_KL
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.counter = 0
        self.min_delta = min_delta
        self.min_validation_loss = np.inf
        self.patience = patience
        self.kl_weight = kl_weight

    def train_step(self,x1,x2,y):
        self.model.train()

        yhat = self.model(x1,x2)
        #print(y.shape)
        loss_mse = self.loss_fn_mse(y, yhat)
        loss_KL = self.loss_fn_KL(self.model)
        
        total_loss = loss_mse + self.kl_weight * loss_KL

        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return total_loss.item()
    
    def earlyStop(self, validation_loss):
        if validation_loss < (self.min_validation_loss - self.min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0
            
        elif validation_loss >= (self.min_validation_loss - self.min_delta):
            self.counter +=1
            if self.counter >= self.patience:
                return True
            return False

    def train(self, train_loader, val_loader, batch_size, n_epochs, n_features, output_dim, mode):

        model_path = f'bayesian_cnn_lstm_{mode}.pt'
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
                    #val_loss = self.train_step(x1_val, x2_val, y_val)
                    val_loss_mse = self.loss_fn_mse(y_val, yhat)
                    val_loss_KL = self.loss_fn_KL(self.model)
                    val_loss = (val_loss_mse + self.kl_weight * val_loss_KL).item()
                    #print(val_loss)
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)
                if self.earlyStop(validation_loss):
                    break_out_flag = True
                    break               
            
            if break_out_flag:
                torch.save(self.model.state_dict(), model_path)
                break

            if (epoch % 10 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )
                #print(
                #    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.6f}"
                #)
                #plt.figure(figsize=[15,5])
                #plt.plot(y_val[-1].cpu(), color = 'black', linestyle = '-')
                #plt.plot(yhat[-1].cpu(), color = 'blue', linestyle = '-')
                #plt.legend(['Actual validation data', 'Network validation results'])
                #plt.title(f"Comparisons after {epoch} epochs")
                #plt.grid()
                #plt.savefig('Validation comparisons after %d epochs.png'%epoch,dpi=300)
                #plt.close()
                
        #model_path = 'cnn_2D_method_example1.pt'
        #model_path = f'dwt_method_for_{training_len}_as_training_length and {decomposition_level} as level and {hidden_dim} as hidden dim and {hidden_dim2} as hidden_dim2.pt'
        torch.save(self.model.state_dict(), model_path)
    
    def evaluate(self, x, test, training_len, output_len, missing_len, sample):
        with torch.no_grad():
            predictions = []
            values = []
            for j in range(len(test)):
                val = test[j].to(device).cpu()
                values.append(val.detach().numpy())
            
            for i in range(missing_len):
                x = x.to(device)
                self.model.eval()
                #x_test = x.view([1, -1, training_len]).to(device)
                x_lstm_test = x.view([1, -1, training_len]).to(device)

                x_a = x.cpu()
                x_arr = np.array(x_a)
                x_arr = x_arr.reshape(len(x_arr))
                #print(x_arr.shape)

                time = (1)*np.arange(0, len(x_arr))
                signal = x_arr
                scales = np.arange(1, 256)

                dt = time[1] - time[0]
                waveletname = 'cmor' 
                cmap = plt.cm.jet
                [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
                power = (abs(coefficients)) ** 2
    
                p_h = 0.001
                p_l = 0.0

                levs = np.arange(p_l,p_h,(p_h - p_l)/100)
    
                fig = plt.figure(figsize=(1,1))
                im = plt.contourf(time, frequencies, power, cmap = cmap, vmin=p_l, vmax=p_h, levels=levs, extend='both')
                plt.axis('off')
                #plt.ylim([0.001, 0.05])
                plt.xlim([0.1*len(x_arr), 0.8*len(x_arr)])
                plt.savefig(f'SamplePred{sample}.jpeg', bbox_inches='tight',pad_inches = 0)
                plt.close()

                image = Image.open(f'SamplePred{sample}.jpeg')

                resized_image = transform2(image)
                tensor = transform(resized_image)
                
                tensor = tensor.unsqueeze(0).to(device)
                #print(tensor.size())

                yhat = self.model(tensor, x_lstm_test)
                yint = torch.reshape(yhat,(output_len,1))
                
                y_int = yint.to(device).cpu()
                
                predictions.append(y_int[-1].detach().numpy())
                x=torch.reshape(x,(training_len,1))
                x = torch.cat((x,yint[-1].reshape(1,1)),0)
                x = x[-training_len:]
            
        preds =  torch.reshape(torch.Tensor(predictions),(-1,1))
        print(preds.shape)
        
        return np.asarray(values), np.asarray(preds)
    
    def evaluate2(self, x, test, training_len, missing_len):
        with torch.no_grad():
            predictions = []
            values = []
            for j in range(len(test)):
                val = test[j].to(device).cpu()
                values.append(val.detach().numpy())
            
            num = len(test) % missing_len
            print(num)
            if (num == 0):    
                for i in range(math.floor(len(test)/missing_len)):
                    x = x.to(device)
                    self.model.eval()
                    #x_test = x.view([1, -1, training_len]).to(device)
                    x_lstm_test = x.view([1, -1, training_len]).to(device)

                    x_a = x.cpu()
                    x_arr = np.array(x_a)
                    x_arr = x_arr.reshape(len(x_arr))
                    #print(x_arr.shape)

                    time =  time = (1)*np.arange(0, len(x_arr))
                    signal = x_arr
                    scales = np.arange(1, 256)

                    dt = time[1] - time[0]
                    waveletname = 'cmor'
                    cmap = plt.cm.jet
                    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
                    power = (abs(coefficients)) ** 2

                    p_h = 0.001
                    p_l = 0.0

                    levs = np.arange(p_l,p_h,(p_h - p_l)/100)

                    fig = plt.figure(figsize=(1,1))
                    im = plt.contourf(time, frequencies, power, cmap = cmap, vmin=p_l, vmax=p_h, levels=levs, extend='both')
                    plt.axis('off')
                    plt.savefig(f'Sample2.jpeg', bbox_inches='tight',pad_inches = 0)
                    plt.close()

                    image = Image.open(f'Sample2.jpeg')

                    resized_image = transform2(image)
                    tensor = transform(resized_image)
                
                    tensor = tensor.unsqueeze(0).to(device)
                    #print(tensor.size())

                    yhat = self.model(tensor, x_lstm_test)
                    yint = torch.reshape(yhat,(missing_len,1))
                
                    y_int = yint.to(device).cpu()
                
                    predictions.append(y_int.detach().numpy())
                    x=torch.reshape(x,(training_len,1))
                    x = torch.cat((x,yint),0)
                    x = x[-training_len:]
                    
            else:
                for i in range(math.floor(len(test)/missing_len)+1):
                    x = x.to(device)
                    self.model.eval()
                    #x_test = x.view([1, -1, training_len]).to(device)
                    x_lstm_test = x.view([1, -1, training_len]).to(device)

                    x_a = x.cpu()
                    x_arr = np.array(x_a)
                    x_arr = x_arr.reshape(len(x_arr))
                    #print(x_arr.shape)

                    time =  time = (1)*np.arange(0, len(x_arr))
                    signal = x_arr
                    scales = np.arange(1, 256)

                    dt = time[1] - time[0]
                    waveletname = 'cmor'
                    cmap = plt.cm.jet
                    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
                    power = (abs(coefficients)) ** 2

                    p_h = 0.001
                    p_l = 0.0

                    levs = np.arange(p_l,p_h,(p_h - p_l)/100)

                    fig = plt.figure(figsize=(1,1))
                    im = plt.contourf(time, frequencies, power, cmap = cmap, vmin=p_l, vmax=p_h, levels=levs, extend='both')
                    plt.axis('off')
                    plt.savefig(f'Sample2.jpeg', bbox_inches='tight',pad_inches = 0)
                    plt.close()

                    image = Image.open(f'Sample2.jpeg')

                    resized_image = transform2(image)
                    tensor = transform(resized_image)
                
                    tensor = tensor.unsqueeze(0).to(device)
                    #print(tensor.size())

                    yhat = self.model(tensor, x_lstm_test)
                    yint = torch.reshape(yhat,(missing_len,1))
                
                    y_int = yint.to(device).cpu()
                
                    predictions.append(y_int.detach().numpy())
                    x=torch.reshape(x,(training_len,1))
                    x = torch.cat((x,yint),0)
                    x = x[-training_len:]
                    
        preds =  torch.reshape(torch.Tensor(predictions),(-1,1))
        
        return np.asarray(preds)


    def plot_losses(self,n_epochs,batch_size,learning_rate, training_len):
        """The method plots the calculated loss values for training and validation
        """
        np.savetxt(f"Training_len={training_len}_train.out", self.train_losses, fmt='%1.4e')
        #np.savetxt(f"Training_len={training_len}_val.out", self.val_losses, fmt='%1.4e')

        plt.figure(figsize=[10,8])
        plt.plot(self.train_losses, label="Training loss")
        #plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title(f"Losses for training length={training_len}")
        plt.grid()
        plt.show()
        #plt.savefig('Losses comparisons for training length=%d.png'%training_len,dpi=300)
        #plt.close()


# In[ ]:


from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import LogSoftmax
import torchbnn as bnn


# In[ ]:


input_dim = input_len
output_dim = output_len

ks = 3
hidden_dim = 128
layer_dim = 2
dropout_prob = 0.3
a_net_feature = 64
mlp_hiddendim1 = 100
mlp_odim = output_len

oc1 = 100

dilation1 = 1
dilation2 = 7

tuple_shape = (77, 77)
weight_decay = 1e-3
ks = 3

learning_rate = 1e-5
batch_size = 32
n_epochs = 1500

Ftrain_loader = DataLoader(trainZ, batch_size=batch_size, shuffle=False, drop_last=True)
Fval_loader = DataLoader(valZ, batch_size=batch_size, shuffle=False, drop_last=True)


# In[1]:


pred_len = test_len
total_len = input_len + pred_len

Test_Matrix_Features = train_val_data[-input_len:]
Test_Matrix_Targets = data[-test_len:]

L = 1

Predictions = np.asarray([[0.0 for x in range(pred_len)] for y in range(L)])
Values = np.asarray([[0.0 for x in range(pred_len)] for y in range(L)])

transform = transforms.ToTensor()
transform2 = transforms.Resize((77,77))

X_Test = np.asarray(Test_Matrix_Features)
Y_Test = np.asarray(Test_Matrix_Targets)

val_int = scaler.inverse_transform(Y_Test.reshape(-1,1))
Values = Y_Test.reshape(len(val_int))
    
Test_features = torch.Tensor(X_Test)
Test_targets = torch.Tensor(Y_Test)


# In[ ]:


for k in range(10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BayesianNet(oc1 = oc1, s_conv = 1, ks = ks, dil1 = dilation1, dil2 = dilation2, ts = tuple_shape, i_dim = input_dim, h_dim = hidden_dim, l_dim = layer_dim, d_prob = dropout_prob, mlp_hdim1 = mlp_hiddendim1, mlp_odim = output_len, a_net_feature = a_net_feature)
    model = model.to(device)

    loss_fn_mse = nn.MSELoss(reduction="mean")
    loss_fn_KL = bnn.BKLLoss(reduction="mean", last_layer_only = False)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    opt = Optimization(model=model, loss_fn_mse=loss_fn_mse, loss_fn_KL=loss_fn_KL, optimizer=optimizer, patience = 20)
    opt.train(Ftrain_loader, Fval_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim, output_dim = output_len, mode = k)
    opt.plot_losses(output_dim)
            
    end = timer()

    dur = (end-start)/60
    print(f'The total duration for the training is {dur} minutes')

    PATH = f"bayesian_cnn_lstm_{k}.pt"

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = BayesianNet(oc1 = oc1, s_conv = 1, ks = ks, dil1 = dilation1, dil2 = dilation2, ts = tuple_shape, i_dim = input_dim, h_dim = hidden_dim, l_dim = layer_dim, d_prob = dropout_prob, mlp_hdim1 = mlp_hiddendim1, mlp_odim = output_len, a_net_feature = a_net_feature)
    #model = model.to(device)

    learning_rate = 1e-5
    batch_size = 1
    n_epochs = 1000

    model.load_state_dict(torch.load(PATH))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    bl1 = Optimization(model=model, loss_fn_mse=loss_fn_mse, loss_fn_KL=loss_fn_KL, optimizer=optimizer, patience = 20)
    preds = bl1.evaluate2(Test_features,Test_targets, input_len, output_len)
    num = len(Test_targets) % output_len

    if (num != 0):
        preds = preds[:len(Test_targets)]
    
    pred_int = scaler.inverse_transform(preds.reshape(-1,1))
    Predictions = pred_int.reshape(len(pred_int))

    np.savetxt(f"twoFreqs_bnn2d+lstm_withAttention_pred_results_{k}.out", Predictions)

