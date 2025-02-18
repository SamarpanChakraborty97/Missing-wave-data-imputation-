import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import Module
from torch.nn import Conv1d
from torch.nn import Linear
from torch.nn import MaxPool1d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten

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

# import the necessary packages
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import LogSoftmax
from torch import flatten
import matplotlib.pyplot as plt

x = np.arange(0,10*math.pi, math.pi/600)
high_freq = 0.1 * np.sin(20*x)
low_freq = np.sin(2*x)
data = np.asarray(high_freq + low_freq)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data = scaler.fit_transform(data.reshape(-1,1))

test_len = int(0.05 * len(data))
train_val_data = data[:-test_len]

#miss_data = data[int(0.5*length)-int(missing_len/2):int(0.5*length)+int(missing_len/2)+1]
#x_miss = x[int(0.5*length)-int(missing_len/2):int(0.5*length)+int(missing_len/2)+1]

input_len = 500
output_len = 1

pred_len = 79
total_len = input_len + pred_len

Test_Matrix_Features = train_val_data[-input_len:]
Test_Matrix_Targets = data[-test_len:]

L = 1

Predictions = np.asarray([[0.0 for x in range(pred_len)] for y in range(L)])
Values = np.asarray([[0.0 for x in range(pred_len)] for y in range(L)])

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
        
        #print("The size of the reshaped convolutional layer has the shape of {}".format(x1_inter.size()))

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
        xd1 = torch.concat((xD1_inter,hn_rec1),dim=2)
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
        
        xdil1 = torch.bmm(torch.transpose(xD1_inter,1,2),xdil1).to(device)
        #xdil2 = torch.bmm(torch.transpose(xD2_inter,1,2),xdil2).to(device)
        
        #print("The size of the input after the weighted sum has the shape of {}".format(x.size()))
        
        xdil1 = xdil1.reshape([x2.size(0),-1])
        #xdil2 = xdil2.reshape([x2.size(0),-1])
        
        #print("The size of the reshaped input layer before MLP has the shape of {}".format(x.size()))

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
    
class Optimization:

    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_step(self,x,y):
        self.model.train()

        yhat = self.model(x)
        #print(y.shape)
        loss = self.loss_fn(y, yhat)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def train(self, train_loader, val_loader, batch_size, n_epochs, n_features, training_len, oc1):

        model_path = f'cnn_2D_using_{oc1}_as_convolutional_features.pt'
        
        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                #x_batch = x_batch.view([batch_size, n_features]).to(device)
                x_batch = x_batch.to(device)
                #print(x_batch.shape)
                y_batch = y_batch.to(device)
                yhat = self.model(x_batch)
                #print(yhat.shape)
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

            if (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )
                
        torch.save(self.model.state_dict(), model_path)
    
    def evaluate(self, x, test, training_len, missing_len):
        with torch.no_grad():
            predictions = []
            values = []
            for i in range(len(test)):
                x = x.to(device)
                self.model.eval()
                x_test = x.view([1, -1, training_len]).to(device)
                
                x_a = x.cpu()
                x_arr = np.array(x_a)
                x_arr = x_arr.reshape(len(x_arr))
                #print(x_arr.shape)
                
                time =  time = (1/48)*np.arange(0, len(x_arr))
                signal = x_arr
                scales = np.arange(1, 512)

                dt = time[1] - time[0]
                waveletname = 'cmor' 
                cmap = plt.cm.jet
                [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
                power = (abs(coefficients)) ** 2
    
                p_h = 4.0
                p_l = 0

                levs = np.arange(p_l,p_h,(p_h - p_l)/100)
    
                fig = plt.figure(figsize=(1,1))
                im = plt.contourf(time, frequencies, power, cmap = cmap, vmin=p_l, vmax=p_h, levels=levs, extend='both')
                plt.ylim([0.046,0.2])
                plt.axis('off')
                plt.savefig(f'Sample.jpeg', bbox_inches='tight',pad_inches = 0)
                plt.close()
                
                image = Image.open(f'Sample.jpeg')
                resized_image = transform2(image)
                tensor = transform(resized_image)
                
                tensor = tensor.unsqueeze(0).to(device)
                #print(tensor.size())
                
                yhat = self.model(tensor)
                yint = torch.reshape(yhat,(missing_len,1,1))
                
                y_int = yint[0].to(device).cpu()
                val = test[i].to(device).cpu()
                
                predictions.append(y_int.detach().numpy())
                values.append(val.detach().numpy())
                print(i)
                #print(yhat)
                #print(x[0:2])
                #print(x[-3:],"\n")
                #print(yint[0].size())
                x=torch.reshape(x,(training_len,1))
                x = torch.cat((x,yint[0]),0)
                x = x[1:]
        preds =  torch.reshape(torch.Tensor(predictions),(len(predictions),1))
        vals =  torch.reshape(torch.Tensor(values),(len(values),1))
        
        return np.asarray(preds), np.asarray(vals)
    
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

    def plot_losses(self,n_epochs,batch_size,learning_rate, training_len, oc1):
        """The method plots the calculated loss values for training and validation
        """
        np.savetxt(f"Convolutional_features={oc1}_train.out", self.train_losses, fmt='%1.4e')
        np.savetxt(f"Convolutional_features={oc1}_val.out", self.val_losses, fmt='%1.4e')

        plt.figure(figsize=[10,8])
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title(f"Losses for convolutional features={oc1}")
        plt.grid()
        plt.savefig('Losses comparisons for convolutional features=%d.png'%oc1,dpi=300)
        plt.close()
        
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

transform = transforms.ToTensor()
transform2 = transforms.Resize((77,77))

X_Test = np.asarray(Test_Matrix_Features)
Y_Test = np.asarray(Test_Matrix_Targets)

val_int = scaler.inverse_transform(Y_Test.reshape(-1,1))
Values = Y_Test.reshape(len(val_int))
    
Test_features = torch.Tensor(X_Test)
Test_targets = torch.Tensor(Y_Test)

PATH = f"cnn_lstm_twoFreqs.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_LSTM_Module(oc1 = oc1, s_conv = 1, ks = ks, dil1 = dilation1, dil2 = dilation2, ts = tuple_shape, i_dim = input_dim, h_dim = hidden_dim, l_dim = layer_dim, d_prob = dropout_prob, mlp_hdim1 = mlp_hiddendim1, mlp_odim = output_len, a_net_feature = a_net_feature)
model = model.to(device)

loss_fn = nn.MSELoss(reduction="mean")

learning_rate = 1e-5
batch_size = 1
n_epochs = 1000

model.load_state_dict(torch.load(PATH))
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fn = nn.MSELoss(reduction="mean")
    
bl1 = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
preds = bl1.evaluate2(Test_features,Test_targets, input_len, output_len)
num = len(Test_targets) % output_len

if (num != 0):
    preds = preds[:len(Test_targets)]

pred_int = scaler.inverse_transform(preds.reshape(-1,1))
Predictions = pred_int.reshape(len(pred_int))

np.savetxt("twoFreqs_cnn2d+lstm_pred_results.out", Predictions)
np.savetxt("twoFreqs_values.out", Values)