#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


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


plt.figure(figsize=[15,5])
plt.plot(x, data, 'k', linewidth = 0.6)


# In[4]:


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


# In[5]:


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


# In[6]:


input_len = 500
output_len = 1
Train_X, Train_Y, Val_X, Val_Y = generateLaggedDf(Eta_Df, input_len, output_len,'Eta')


# In[7]:


import torch
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import datetime as datetime
from timeit import default_timer as timer
import pywt


# In[8]:


X_Train = np.asarray(Train_X.iloc[:,::-1])
Y_Train = np.asarray(Train_Y.iloc[:,::-1])
X_Val = np.asarray(Val_X.iloc[:,::-1])
Y_Val = np.asarray(Val_Y.iloc[:,::-1])


# In[9]:


X_t = torch.Tensor(X_Train.copy())
X_v = torch.Tensor(X_Val.copy())


# In[10]:


import torchvision
import torchvision.transforms as transforms
from PIL import Image
transform = transforms.ToTensor()
import math


# In[11]:


X_Train_LSTM_Tensor = torch.zeros(len(X_Train), input_len)
X_Val_LSTM_Tensor = torch.zeros(len(X_Val), input_len)


# In[12]:


for i in range(len(X_Train_LSTM_Tensor)):
    X_Train_LSTM_Tensor[i] = X_t[i]

for j in range(len(X_Val_LSTM_Tensor)):
    X_Val_LSTM_Tensor[j] = X_v[j]


# In[13]:


Y_t = torch.Tensor(Y_Train.copy())
Y_v = torch.Tensor(Y_Val.copy())

Y_Train_Tensor = torch.zeros(len(Y_Train),output_len)
Y_Val_Tensor = torch.zeros(len(Y_Val),output_len)


# In[14]:


for i in range(len(Y_Train_Tensor)):    
    Y_Train_Tensor[i] = Y_t[i]

for j in range(len(Y_Val_Tensor)):
    Y_Val_Tensor[j] = Y_v[j]


# In[ ]:


X_Train_CNN_Tensor = torch.zeros(len(X_Train), 3 , 77, 77)

for k in range(len(X_Train)):
    time = (1)*np.arange(0, len(X_Train[k]))
    signal = X_Train[k]
    scales = np.arange(1, 256)

    dt = time[1] - time[0]
    waveletname = 'cmor' 
    cmap = plt.cm.jet
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2
    
    p_h = 0.001
    p_l = 0

    levs = np.arange(p_l,p_h,(p_h - p_l)/100)
    
    fig = plt.figure(figsize=(1,1))
    im = plt.contourf(time, frequencies, power, cmap = cmap, vmin=p_l, vmax=p_h, levels=levs, extend='both')
    plt.axis('off')
    #plt.ylim([0.001, 0.05])
    plt.xlim([0.1*len(X_Train[0]), 0.8*len(X_Train[0])])
    plt.savefig(f'Sample1.jpeg', bbox_inches='tight',pad_inches = 0)
    plt.close()
    
    image = Image.open(f'Sample1.jpeg')
    tensor = transform(image)
    X_Train_CNN_Tensor[k] = tensor


X_Val_CNN_Tensor = torch.zeros(len(X_Val), 3 , 77, 77)

for k in range(len(X_Val)):
    time = (1)*np.arange(0, len(X_Val[k]))
    signal = X_Val[k]
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
    plt.xlim([0.1*len(X_Val[0]), 0.8*len(X_Val[0])])
    plt.savefig(f'SampleVal1.jpeg', bbox_inches='tight',pad_inches = 0)
    plt.close()
    
    image = Image.open(f'SampleVal1.jpeg')
    tensor = transform(image)
    X_Val_CNN_Tensor[k] = tensor


# In[ ]:


torch.save(X_Train_LSTM_Tensor, 'X_Train_LSTM_Tensor.pt')
torch.save(Y_Train_Tensor, 'Y_Train_Tensor.pt')
torch.save(X_Train_CNN_Tensor, 'X_CNN_LSTM_Train_Tensor.pt')
    
torch.save(X_Val_LSTM_Tensor, 'X_Val_LSTM_Tensor.pt')
torch.save(Y_Val_Tensor, 'Y_Val_Tensor.pt')
torch.save(X_Val_CNN_Tensor, 'X_CNN_LSTM_Val_Tensor.pt')

