#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Z_displacement2.csv")
x = data.index
y = data['Eta']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

req_data = y[0:20000]
Full_TimeSeries = req_data
data_dum = scaler.fit_transform(np.asarray(Full_TimeSeries).reshape(-1,1))


L = len(Full_TimeSeries)
missing_len = int(0.1*L)
data_new = scaler.fit_transform(np.asarray(Full_TimeSeries).reshape(-1,1))
data_new[int(0.5*L)-int(missing_len/2):int(0.5*L)+int(missing_len/2)].fill(np.NaN)
Eta_Df = pd.DataFrame([])
df = pd.DataFrame(data_new, columns = ['Eta'])
Eta_Df = Eta_Df.append(df, ignore_index = True)


Actual_Missing_Data = data_dum[9000:11000]
Actual_Training_Data1 = data_dum[:8999]
Actual_Training_Data2 = data_dum[11001:]

Act_TrainData = np.append(Actual_Training_Data1, Actual_Training_Data2)

sigma = np.std(Act_TrainData)

def generateLaggedDf(df, training_len, missing_len, col_name):
    df_new = df.copy()
    for i in range(1,training_len + missing_len):
        df_new[f"Lag{i}"] = df[[col_name]].shift(i)
    df_new = df_new.iloc[training_len + missing_len:]
    
    df_new = df_new.dropna(axis= 0)
    
    mid_df = int(0.5 * len(df_new))
    train_len2 = int(0.7 * 0.5 * len(df_new))
    
    df_train1 = df_new[:train_len2]
    df_val1 = df_new[train_len2:mid_df]
    
    df_train2 = df_new[mid_df:mid_df+train_len2]
    df_val2 = df_new[mid_df+train_len2:]
    
    df_train = df_train1.append(df_train2, ignore_index = True)
    df_val = df_val1.append(df_val2, ignore_index = True)
    
    trainY = df_train.iloc[:,:missing_len]
    trainX = df_train.drop(df_train.iloc[:,:missing_len], axis=1)
    
    valY = df_val.iloc[:,:missing_len]
    valX = df_val.drop(df_train.iloc[:,:missing_len], axis=1)
    
    return trainX, trainY, valX, valY


input_len = 500
output_len = 1
Train_X, Train_Y, Val_X, Val_Y = generateLaggedDf(Eta_Df, input_len, output_len,'Eta')


X_Train = np.asarray(Train_X.iloc[:,::-1])
Y_Train = np.asarray(Train_Y.iloc[:,::-1])
X_Val = np.asarray(Val_X.iloc[:,::-1])
Y_Val = np.asarray(Val_Y.iloc[:,::-1])


# In[16]:


import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import datetime as datetime
from timeit import default_timer as timer

torch.manual_seed(200)

Train_features = torch.Tensor(X_Train.copy())
Train_targets = torch.Tensor(Y_Train.copy())
Val_features = torch.Tensor(X_Val.copy())
Val_targets = torch.Tensor(Y_Val.copy())

trainZ = TensorDataset(Train_features, Train_targets)
valZ = TensorDataset(Val_features, Val_targets)

import torch.nn as nn

X_Train = np.asarray(Train_X.iloc[:,::-1])
Y_Train = np.asarray(Train_Y.iloc[:,::-1])
X_Val = np.asarray(Val_X.iloc[:,::-1])
Y_Val = np.asarray(Val_Y.iloc[:,::-1])

import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import datetime as datetime
from timeit import default_timer as timer

torch.manual_seed(200)

Train_features = torch.Tensor(X_Train.copy())
Train_targets = torch.Tensor(Y_Train.copy())
Val_features = torch.Tensor(X_Val.copy())
Val_targets = torch.Tensor(Y_Val.copy())

train = TensorDataset(Train_features, Train_targets)
val = TensorDataset(Val_features, Val_targets)

import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim,  output_dim, dropout_prob, hidden_dim, layer_dim):
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out

def get_model(model, model_params):
    models = {
        #"rnn": RNNModel,
        "lstm": LSTMModel,
        #"bi-lstm": BiLSTMModel,
        #"gru": GRUModel,
    }
    return models.get(model.lower())(**model_params)

class Optimization:
    """Optimization is a helper class that allows training, validation, prediction.
    """
    def __init__(self, model, optimizer, min_delta = 1e-6):

        self.model = model
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.counter = 0
        self.min_delta = min_delta
        self.min_validation_loss = np.inf
        
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

    def train(self, train_loader, val_loader, batch_size, n_epochs, n_features, output_dim):

        model_path = f'lstm_with_output_length_{output_dim}.pt'
        break_out_flag = False

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
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
            #torch.save(self.model.state_dict(), model_path)    
            #tune.report(validation_loss)
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
    
    def evaluate2(self, x, test, training_len, missing_len):
        with torch.no_grad():
            predictions = []
            values = []
            for j in range(len(test)):
                val = test[j].to(device).cpu()
                values.append(val.detach().numpy())
            
            num = len(test) % missing_len
            if (num == 0):
                for i in range(math.floor(len(test)/missing_len)):
                    x = x.to(device)
                    self.model.eval()
                    x_test = x.view([1, -1, training_len]).to(device)
                
                    yhat = self.model(x_test)
                    yint = torch.reshape(yhat,(missing_len,1))                
                    y_int = yint.to(device).cpu()
                    predictions.append(y_int.detach().numpy())
                    x = torch.reshape(x,(training_len,1))
                    x = torch.cat((x,yint),0)
                    x = x[-training_len:]
            else:
                for i in range(math.floor(len(test)/missing_len)+1):
                    x = x.to(device)
                    self.model.eval()
                    x_test = x.view([1, -1, training_len]).to(device)
                
                    yhat = self.model(x_test)
                    yint = torch.reshape(yhat,(missing_len,1))                
                    y_int = yint.to(device).cpu()
                    predictions.append(y_int.detach().numpy())
                    x = torch.reshape(x,(training_len,1))
                    x = torch.cat((x,yint),0)
                    x = x[-training_len:]
            
        preds =  torch.reshape(torch.Tensor(predictions),(-1,1))
        return np.asarray(preds)

    def plot_losses(self, training_len):
        """The method plots the calculated loss values for training and validation
        """
        np.savetxt(f"Output_length={training_len}_train.out", self.train_losses, fmt='%1.4e')
        np.savetxt(f"Output_length={training_len}_val.out", self.val_losses, fmt='%1.4e')
        
        plt.figure(figsize=[10,8])
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title(f"Losses for output length = {training_len}")
        plt.grid()
        plt.show()
        plt.savefig(f'Losses comparisons for output length={training_len} over epochs.png',dpi=300)
        plt.close()

class NormalLoss(nn.Module):
    def  __init__(self, preds, labels, sig):
        super(NormalLoss, self).__init__()
        self.preds = preds
        self.labels = labels
        self.sig = sig
    
    def forward(self, preds, labels, sig):
        mu = preds
        log_sigma = sig
        log_likelihood = Normal(mu, log_sigma.exp()).log_prob(labels)
        
        return -log_likelihood.sum()

class Optimization2:
    """Optimization is a helper class that allows training, validation, prediction.
    """
    def __init__(self, model, optimizer, patience, min_delta = 1e-6, sig = sigma):  

        self.model = model
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.counter = 0
        self.min_delta = min_delta
        self.min_validation_loss = np.inf
        self.patience = patience
        self.sig = sig
        
    def computeLoss(self, preds, labels, sig):
        mu = preds
        log_sigma = torch.Tensor([sig]).to(device)
        log_likelihood = Normal(mu, log_sigma.exp()).log_prob(labels)
        
        return -log_likelihood.sum()
        
    def train_step(self, x, y):

        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.computeLoss(yhat, y, sigma)

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

    def train(self, train_loader, val_loader, batch_size, n_epochs, n_features, output_dim):

        model_path = f'lstm_with_output_length_{output_dim}_MLE.pt'
        break_out_flag = False

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.computeLoss(yhat, y_val, sigma).item()
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
            #torch.save(self.model.state_dict(), model_path)    
            #tune.report(validation_loss)
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
    
    def evaluate2(self, x, test, training_len, missing_len):
        with torch.no_grad():
            predictions = []
            values = []
            for j in range(len(test)):
                val = test[j].to(device).cpu()
                values.append(val.detach().numpy())
            
            num = len(test) % missing_len
            if (num == 0):
                for i in range(math.floor(len(test)/missing_len)):
                    x = x.to(device)
                    self.model.eval()
                    x_test = x.view([1, -1, training_len]).to(device)
                
                    yhat = self.model(x_test)
                    yint = torch.reshape(yhat,(missing_len,1))                
                    y_int = yint.to(device).cpu()
                    predictions.append(y_int.detach().numpy())
                    x = torch.reshape(x,(training_len,1))
                    x = torch.cat((x,yint),0)
                    x = x[-training_len:]
            else:
                for i in range(math.floor(len(test)/missing_len)+1):
                    x = x.to(device)
                    self.model.eval()
                    x_test = x.view([1, -1, training_len]).to(device)
                
                    yhat = self.model(x_test)
                    yint = torch.reshape(yhat,(missing_len,1))                
                    y_int = yint.to(device).cpu()
                    predictions.append(y_int.detach().numpy())
                    x = torch.reshape(x,(training_len,1))
                    x = torch.cat((x,yint),0)
                    x = x[-training_len:]
            
        preds =  torch.reshape(torch.Tensor(predictions),(-1,1))


    def plot_losses(self, training_len):
        """The method plots the calculated loss values for training and validation
        """
        np.savetxt(f"Output_length={training_len}_train.out", self.train_losses, fmt='%1.4e')
        np.savetxt(f"Output_length={training_len}_val.out", self.val_losses, fmt='%1.4e')
        
        plt.figure(figsize=[10,8])
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title(f"Losses for output length = {training_len}")
        plt.grid()
        plt.show()
        plt.savefig(f'Losses comparisons for output length={training_len} over epochs.png',dpi=300)
        plt.close()


from torch.distributions import Normal
import math

start = timer()
input_dim = input_len
output_dim = output_len
hidden_dim = 32
layer_dim = 2
dropout = 0
weight_decay = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_params = {'input_dim': input_dim,
                'hidden_dim' : hidden_dim,
                'layer_dim' : layer_dim,
                'output_dim' : output_dim,
                'dropout_prob' : dropout}

model = get_model('lstm', model_params)
model = model.to(device)

learning_rate = 1e-4
batch_size = 32
n_epochs = 1000

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
Ftrain_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
Fval_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)

opt = Optimization2(model=model, optimizer=optimizer, patience = 80, sig = sigma)
opt.train(Ftrain_loader, Fval_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim, output_dim = output_dim)
opt.plot_losses(output_dim)
            
end = timer()

dur = (end-start)/60
print(f'The total duration for the training is {dur} minutes')
