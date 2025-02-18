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

    
class Optimization:

    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_step(self,x1,x2,y):
        self.model.train()

        yhat = self.model(x1,x2)
        #print(y.shape)
        loss = self.loss_fn(y, yhat)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def train(self, train_loader, val_loader, batch_size, n_epochs, n_features, training_len):

        model_path = f'cnn+lstm_using_{training_len}_as_training_length.pt'

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x1_batch, x2_batch, y_batch in train_loader:
                x2_batch = x2_batch.view([batch_size, -1, n_features]).to(device)
                #x1_batch = x1_batch.to(device)
                x1_batch = x1_batch.to(device)
                #print(x_batch.shape)
                y_batch = y_batch.to(device)
                yhat = self.model(x1_batch, x2_batch)
                #print(yhat.shape)
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

            if (epoch % 10 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )
        torch.save(self.model.state_dict(), model_path)

    def evaluate(self, x, test, training_len, missing_len, scales2):
        with torch.no_grad():
            predictions = []
            values = []
            #for i in range(len(test)):
            for i in range(len(test)):
                x = x.to(device)
                self.model.eval()
                #x_test = x.view([1, -1, training_len]).to(device)
                x_lstm_test = x.view([1, -1, training_len]).to(device)

                x_a = x.cpu()
                x_arr = np.array(x_a)
                x_arr = x_arr.reshape(len(x_arr))
                #print(x_arr.shape)

                time =  time = (1/48)*np.arange(0, len(x_arr))
                signal = x_arr
                scales = np.arange(1, scales2)

                dt = time[1] - time[0]
                waveletname = 'cmor'
                cmap = plt.cm.jet
                [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
                power = (abs(coefficients)) ** 2

                p_h = np.max(power)
                p_l = np.min(power)

                levs = np.arange(p_l,p_h,(p_h - p_l)/100)

                fig = plt.figure(figsize=(2,2))
                im = plt.contourf(time, frequencies, power, cmap = cmap, vmin=p_l, vmax=p_h, levels=levs, extend='both')
                plt.ylim([0.046,0.2])
                plt.axis('off')
                plt.savefig(f'Sample.jpeg', bbox_inches='tight',pad_inches = 0)
                plt.close()

                image = Image.open(f'Sample.jpeg')

                tensor = transform(image)
    
                #tensor = transform(image)
                
                tensor = tensor.unsqueeze(0).to(device)
                #print(tensor.size())

                yhat = self.model(tensor, x_lstm_test)
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
                    x_test = x.view([1, -1, training_len]).to(device)
                
                    x_a = x.cpu()
                    x_arr = np.array(x_a)
                    x_arr = x_arr.reshape(len(x_arr))
                
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
                    plt.savefig(f'Sample.jpeg', bbox_inches='tight',pad_inches = 0)
                    plt.close()
                
                    image = Image.open(f'Sample.jpeg')
                    resized_image = transform2(image)
                    tensor = transform(resized_image)
                
                    tensor = tensor.unsqueeze(0).to(device)
                    #print(tensor.size())
                
                    yhat = self.model(tensor)
                    #print(yhat.size())
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
                    x_test = x.view([1, -1, training_len]).to(device)
                
                    x_a = x.cpu()
                    x_arr = np.array(x_a)
                    x_arr = x_arr.reshape(len(x_arr))
                
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
                    plt.savefig(f'Sample.jpeg', bbox_inches='tight',pad_inches = 0)
                    plt.close()
                
                    image = Image.open(f'Sample.jpeg')
                    resized_image = transform2(image)
                    tensor = transform(resized_image)
                
                    tensor = tensor.unsqueeze(0).to(device)
                    #print(tensor.size())
                
                    yhat = self.model(tensor)
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
        np.savetxt(f"Training_len={training_len}_val.out", self.val_losses, fmt='%1.4e')

        plt.figure(figsize=[10,8])
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title(f"Losses for training length={training_len}")
        plt.grid()
        plt.savefig('Losses comparisons for training length=%d.png'%training_len,dpi=300)
        plt.close()
        
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

loss_fn = nn.MSELoss(reduction="mean")

learning_rate = 1e-5
batch_size = 1
n_epochs = 1500

transform = transforms.ToTensor()
transform2 = transforms.Resize((77,77))

X_Test = np.asarray(Test_Matrix_Features)
Y_Test = np.asarray(Test_Matrix_Targets)
val_int = scaler.inverse_transform(Y_Test.reshape(-1,1))
Values = Y_Test.reshape(len(val_int))
    
Test_features = torch.Tensor(X_Test)
Test_targets = torch.Tensor(Y_Test)

PATH = f"cnn_2d_twoFrequencies.pt"

model = CNN_Module(out_features = output_dim, ts = tuple_shape, s_conv = 1,  oc1 = oc1, ks = ks, dil1 = dilation1, dil2 = dilation2, i_dim = i_dim, d_prob = dropout)
model = model.to(device)

model.load_state_dict(torch.load(PATH))
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fn = nn.MSELoss(reduction="mean")
    
bl1 = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
preds = bl1.evaluate2(Test_features,Test_targets, input_len, output_len)
num = len(Test_targets) % output_len

if (num != 0):
    preds = preds[:len(Test_targets)]

pred_int = scaler.inverse_transform(preds.reshape(-1,1))
Predictions = preds.reshape(len(pred_int))

np.savetxt("twoFreqs_cnn2d_pred_results.out", Predictions)
np.savetxt("twoFreqs_values.out", Values)