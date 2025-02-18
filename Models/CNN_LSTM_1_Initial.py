#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

import torch.nn as nn
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
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

transform = transforms.ToTensor()
transform2 = transforms.Resize((54,55))


# In[3]:


for i in range(1,3):
    data_pre = pd.read_csv(f"Slow_amp_pre_{i}.csv", header=None)
    data_post = pd.read_csv(f"Slow_amp_post_{i}.csv", header=None)
    data_whole = pd.read_csv(f"Slow_amp_whole_{i}.csv", header=None)
    
    n_rows = data_pre.shape[0]
    n_cols = data_whole.shape[1] - (data_pre.shape[1] + data_post.shape[1])

    data_miss = pd.DataFrame(np.zeros([n_rows, n_cols])*np.nan)
    
    data_pre_vals = data_pre[:].values
    data_post_vals = data_post[:].values
    data_whole_vals = data_whole[:].values
    
    data_test = scaler.fit_transform(data_whole_vals.reshape(-1,1)).reshape(data_whole_vals.shape[0],data_whole_vals.shape[1])
    missing_len = data_miss.shape[1]
    
    #Predictions_pre = np.zeros([data_miss.shape[0], data_miss.shape[1]])
    #Values = np.zeros([data_miss.shape[0], data_miss.shape[1]])

    dummy_data = pd.concat([data_pre, data_post], axis=1,ignore_index=True)
    data = scaler.fit_transform(dummy_data[:].values.reshape(-1,1)).reshape(dummy_data.shape[0],dummy_data.shape[1])
    pre_data_scaled = data[:,:data_pre.shape[1]]
    pre_shape = pre_data_scaled.shape
    #data = data[:3,-254:]
    
    input_len = 200
    output_len = 1  
    input_dim = input_len
    output_dim = output_len
    tuple_shape = (53, 54)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    L1 = data.shape[1] - input_len
    train_len = int(0.7 * L1)
    val_len = L1 - train_len
    
    X_train_pre_lstm_tensor = torch.zeros(len(data), train_len, input_len)
    Y_train_pre_tensor = torch.zeros(len(data), train_len, output_len)
    X_train_pre_CNN_tensor = torch.zeros(len(data), train_len, 3 , 53, 54)
    
    
    X_val_pre_lstm_tensor = torch.zeros(len(data), val_len, input_len)
    Y_val_pre_tensor = torch.zeros(len(data), val_len, output_len)
    X_val_pre_CNN_tensor = torch.zeros(len(data), val_len, 3 , 53, 54)

    for j in range(len(data)):
        X_pre = np.zeros([L1, input_len])
        Y_pre = np.zeros([L1, output_len])

        for k in range(L1):
            X_pre[k,:] = data[j,k:k+input_len]
            Y_pre[k,:] = data[j,k+input_len:k+input_len+output_len]

        Train_X_pre = X_pre[:train_len]
        Train_Y_pre = Y_pre[:train_len]
        
        Val_X_pre = X_pre[train_len:]        
        Val_Y_pre = Y_pre[train_len:]
    
        X_train_pre_lstm_tensor[j] = torch.Tensor(Train_X_pre.copy())
        Y_train_pre_tensor[j] = torch.Tensor(Train_Y_pre.copy())
    
        X_val_pre_lstm_tensor[j] = torch.Tensor(Val_X_pre.copy())
        Y_val_pre_tensor[j] = torch.Tensor(Val_Y_pre.copy())
        
        for m in range(train_len):            
            time = (1/1.28)*np.arange(0, len(Train_X_pre[m]))
            signal = Train_X_pre[m]
            scales = np.arange(1, 256)
            dt = time[1] - time[0]
            waveletname = 'cmor' 
            cmap = plt.cm.jet
            [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
            power = (abs(coefficients)) ** 2    
            p_h = 0.05
            p_l = 0.0
            levs = np.arange(p_l,p_h,(p_h - p_l)/100)    
            fig = plt.figure(figsize=(0.7,0.7))
            im = plt.contourf(time, frequencies, power, cmap = cmap, vmin=p_l, vmax=p_h, levels=levs, extend='both')
            plt.axis('off')
            plt.xlim([20, 150])
            plt.ylim([0.005, 0.04])
            plt.savefig(f'Sample{i}.jpeg', bbox_inches='tight',pad_inches = 0)
            plt.close()
            #print(f'm:{m}')
        
            image = Image.open(f'Sample{i}.jpeg')
            tensor = transform(image)
            X_train_pre_CNN_tensor[j,m] = tensor

        for m in range(val_len) :  
            time = (1/1.28)*np.arange(0, len(Val_X_pre[m]))
            signal = Val_X_pre[m]
            scales = np.arange(1, 256)
            dt = time[1] - time[0]
            waveletname = 'cmor' 
            cmap = plt.cm.jet
            [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
            power = (abs(coefficients)) ** 2   
            p_h = 0.05
            p_l = 0.0
            levs = np.arange(p_l,p_h,(p_h - p_l)/100)    
            fig = plt.figure(figsize=(0.7,0.7))
            im = plt.contourf(time, frequencies, power, cmap = cmap, vmin=p_l, vmax=p_h, levels=levs, extend='both')
            plt.axis('off')
            plt.xlim([20, 150])
            plt.ylim([0.005, 0.04])
            plt.savefig(f'SampleVal{i}.jpeg', bbox_inches='tight',pad_inches = 0)
            plt.close()
        
            image = Image.open(f'SampleVal{i}.jpeg')
            tensor = transform(image)
            X_val_pre_CNN_tensor[j,m] = tensor
            
        #print(f'j:{j}')
            
    torch.save(X_train_pre_lstm_tensor, f'X_Train_LSTM_Tensor_{i}.pt')
    torch.save(Y_train_pre_tensor, f'Y_Train_Tensor_{i}.pt')
    torch.save(X_train_pre_CNN_tensor, f'X_CNN_LSTM_Train_Tensor_{i}.pt')
    
    torch.save(X_val_pre_lstm_tensor, f'X_Val_LSTM_Tensor_{i}.pt')
    torch.save(Y_val_pre_tensor, f'Y_Val_Tensor_{i}.pt')
    torch.save(X_val_pre_CNN_tensor, f'X_CNN_LSTM_Val_Tensor_{i}.pt')

