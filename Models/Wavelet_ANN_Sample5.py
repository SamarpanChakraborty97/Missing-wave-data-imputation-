#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
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


# In[3]:


Full_TimeSeries = [y[:577],y[580:2950],y[2954:4077],y[4080:4731],y[4734:5865],y[5869:7423],y[7426:14442],y[14445:14910],y[14913:19719],y[19722:20935],y[20942:26367]]


# In[4]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[5]:


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


# In[6]:


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


# In[7]:


input_len = 800
output_len = 1
Train_X, Train_Y, Val_X, Val_Y = generateLaggedDf(Eta_Df, input_len, output_len,'Eta')


# In[8]:


X_Train = np.asarray(Train_X.iloc[:,::-1])
Y_Train = np.asarray(Train_Y.iloc[:,::-1])
X_Val = np.asarray(Val_X.iloc[:,::-1])
Y_Val = np.asarray(Val_Y.iloc[:,::-1])


# In[9]:


import pywt
level = 4
rand_X = X_Train[7]
coeffs = pywt.wavedec(rand_X, 'db5', level = level)

features = 0
for i in range(len(coeffs)):
    features += len(coeffs[i])


# In[10]:


samples_train = X_Train.shape[0]
samples_val = X_Val.shape[0]


# In[11]:


Train_DWT = np.asarray([[0.0 for x in range(features)] for y in range(samples_train)])
Val_DWT = np.asarray([[0.0 for x in range(features)] for y in range(samples_val)])

for i in range(samples_train):
    arr = []
    coeffs = pywt.wavedec(X_Train[i], 'db5', level = level)
    for j in range(level+1):
        arr = np.append(arr, coeffs[j])
    Train_DWT[i,:] = arr
    
for i in range(samples_val):
    arr = []
    coeffs = pywt.wavedec(X_Val[i], 'db5', level = level)
    for j in range(level+1):
        arr = np.append(arr, coeffs[j])
    Val_DWT[i,:] = arr


# In[12]:


import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import datetime as datetime
from timeit import default_timer as timer


# In[13]:


torch.manual_seed(200)

Train_features = torch.Tensor(Train_DWT.copy())
Train_targets = torch.Tensor(Y_Train.copy())
Val_features = torch.Tensor(Val_DWT.copy())
Val_targets = torch.Tensor(Y_Val.copy())

trainZ = TensorDataset(Train_features, Train_targets)
valZ = TensorDataset(Val_features, Val_targets)


# In[14]:


import torch.nn as nn


# In[15]:


class MLP(nn.Module):
    def __init__(self, input_size,  output_size, hidden1_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.fc1 = nn.Linear(self.input_size, self.hidden1_size)
        self.hidden2_size = math.floor(0.8 * hidden1_size)
        self.fc2 = nn.Linear(self.hidden1_size, self.hidden2_size)
        self.output_size = output_size
        self.fc3 = nn.Linear(self.hidden2_size, self.output_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        hidden_state_1 = self.fc1(x)
        relu1 = self.relu(hidden_state_1)
        hidden_state_2 = self.fc2(relu1)
        relu2 = self.relu(hidden_state_2)
        output = self.fc3(relu2)
        #output = self.tanh(output)
        
        return output


# In[16]:


class Optimization:
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
        
    def train_step(self,x,y):
        self.model.train()
        
        yhat = self.model(x)
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
    
    def train(self, train_loader, val_loader, batch_size, n_epochs, n_features, training_len):
        
        break_out_flag = False
        
        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, n_features]).to(device)
                y_batch = y_batch.to(device)
                yhat = self.model(x_batch)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)
            
            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, n_features]).to(device)
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
                
        model_path = f'dwt_method_for_{training_len} as training length.pt'
        torch.save(self.model.state_dict(), model_path)
    
    def evaluate(self, x, test, training_len, missing_len):
        with torch.no_grad():
            predictions = []
            values = []
            for i in range(len(test)):
                x = x.to(device)
                self.model.eval()
                x_test = x.view([1, -1, training_len]).to(device)
                
                arr = []
                coeffs = pywt.wavedec(np.asarray(x_test.cpu()), 'db5', level = 4)
                for j in range(level+1):
                    arr = np.append(arr, coeffs[j])
                arr_T = torch.Tensor(arr).to(device)
                
                yhat = self.model(arr_T)
                yint = torch.reshape(yhat,(missing_len,1,1))
                
                y_int = yint[0].to(device).cpu()
                val = test[i].to(device).cpu()
                
                predictions.append(y_int.detach().numpy())
                values.append(val.detach().numpy())
                #print(yhat)
                #print(x.size())
                #print(yint[0].size())
                x=torch.reshape(x,(training_len,1))
                x = torch.cat((x,yint[0]),0)
                x = x[1:]
        preds =  torch.reshape(torch.Tensor(predictions),(len(predictions),1))
        vals =  torch.reshape(torch.Tensor(values),(len(values),1))
        
        return np.asarray(preds), np.asarray(vals)
     
    def plot_losses(self, training_len):
        """The method plots the calculated loss values for training and validation
        """
    
        np.savetxt(f"Training_len={training_len}_train.out", self.train_losses, fmt='%1.5e')
        np.savetxt(f"Training_len={training_len}_val.out", self.val_losses, fmt='%1.5e')
        
        plt.figure(figsize=[10,8])
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title(f"Losses for training len = {training_len}")
        plt.grid()
        plt.show()
        plt.savefig(f'Losses comparisons for training len={training_len} over epochs.png',dpi=300)
        plt.close()


# In[19]:


start = timer()

input_dim = features
output_dim = output_len
training_len = input_len
weight_decay = 1e-6
hidden_dim = 150

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(input_dim, output_dim, hidden_dim).to(device)
model = model.to(device)

loss_fn = nn.MSELoss(reduction="mean")

learning_rate = 0.000132
batch_size = 32
n_epochs = 1000

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
Ftrain_loader = DataLoader(trainZ, batch_size=batch_size, shuffle=False, drop_last=True)
Fval_loader = DataLoader(valZ, batch_size=batch_size, shuffle=False, drop_last=True)

opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer, patience = 10)
opt.train(Ftrain_loader, Fval_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim, training_len=training_len)
opt.plot_losses(training_len)
            
end = timer()

dur = (end-start)/60
print(f'The total duration for the training is {dur} minutes')
