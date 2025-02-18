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
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import datetime as datetime
from timeit import default_timer as timer


# In[2]:


class LSTMModel(nn.Module):
    def __init__(self, input_dim,  output_dim, dropout_prob, hidden_dim, layer_dim):
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.d_prob = dropout_prob

        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob)
        
        # Dropout Layer
        self.dropout = nn.Dropout(self.d_prob)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = out[:, -1, :]
        
        out = self.dropout(out)

        out = self.fc(out)

        return out


# In[3]:


def generateLaggedDf(df, training_len, missing_len, col_name):
    df_new = df.copy()
    for i in range(1,training_len + missing_len):
        df_new[f"Lag{i}"] = df[[col_name]].shift(i)
    df_new = df_new.iloc[training_len + missing_len:]
    
    df_new = df_new.dropna(axis= 0)
    
    mid_df = int(0.5 * len(df_new))
    
    df_1 = df_new[:mid_df]
    df_2 = df_new[mid_df:]
   
    train_len1 = int(0.8 * len(df_1))
    train_len2 = int(0.8 * len(df_2))
    
    df_train1 = df_1[:train_len1]
    df_val1 = df_1[train_len1:]
    
    df_train2 = df_2[:train_len2]
    df_val2 = df_2[train_len2:]
    
    df_train = df_train1.append(df_train2, ignore_index = True)
    df_val = df_val1.append(df_val2, ignore_index = True)
    
    trainY = df_train.iloc[:,:missing_len]
    trainX = df_train.drop(df_train.iloc[:,:missing_len], axis=1)
    
    valY = df_val.iloc[:,:missing_len]
    valX = df_val.drop(df_train.iloc[:,:missing_len], axis=1)
    
    return trainX, trainY, valX, valY


# In[4]:


class Optimization:
    """Optimization is a helper class that allows training, validation, prediction.
    """
    def __init__(self, model, loss_fn, optimizer, patience, min_delta = 1e-6):

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

    def train(self, train_loader, val_loader, batch_size, n_epochs, mode, n_features, output_dim):

        model_path = f'lstm_1.pt'
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

            #if (epoch % 50 == 0):
            #    print(
            #        f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
            #    )
        #torch.save(self.model.state_dict(), model_path)
          
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
        
        return np.asarray(values), np.asarray(preds)


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


# In[5]:


def get_model(model, model_params):
    models = {
        #"rnn": RNNModel,
        "lstm": LSTMModel,
        #"bi-lstm": BiLSTMModel,
        #"gru": GRUModel,
    }
    return models.get(model.lower())(**model_params)


# In[6]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[7]:


for i in range(1,21):
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
    
    Predictions_pre = np.zeros([data_miss.shape[0], data_miss.shape[1]])
    Values = np.zeros([data_miss.shape[0], data_miss.shape[1]])

    dummy_data = pd.concat([data_pre, data_post], axis=1,ignore_index=True)

    data = scaler.fit_transform(dummy_data[:].values.reshape(-1,1)).reshape(dummy_data.shape[0],dummy_data.shape[1])

    pre_data_scaled = data[:,:data_pre.shape[1]]

    pre_shape = pre_data_scaled.shape
    
    input_len = 200
    output_len = 1

    input_dim = input_len
    output_dim = output_len
    hidden_dim = 200
    layer_dim = 2
    dropout = 0.3
    weight_decay = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    L1 = data.shape[1] - input_len

    for j in range(len(data)):
    #for j in range(2):
        train_len1 = int(0.7 * L1)

        X_pre = np.zeros([L1, input_len])
        Y_pre = np.zeros([L1, output_len])

        for k in range(L1):
            X_pre[k,:] = data[j,k:k+input_len]
            Y_pre[k,:] = data[j,k+input_len:k+input_len+output_len]

        Train_X_pre = X_pre[:train_len1]
        Train_Y_pre = Y_pre[:train_len1]
        
        Val_X_pre = X_pre[train_len1:]        
        Val_Y_pre = Y_pre[train_len1:]
    
        X_train_pre_tensor = torch.Tensor(Train_X_pre.copy())
        Y_train_pre_tensor = torch.Tensor(Train_Y_pre.copy())
    
        X_val_pre_tensor = torch.Tensor(Val_X_pre.copy())
        Y_val_pre_tensor = torch.Tensor(Val_Y_pre.copy())
    
        mode = j
    
        torch.manual_seed(2)
    
        train_pre_eta = TensorDataset(X_train_pre_tensor, Y_train_pre_tensor)
        val_pre_eta = TensorDataset(X_val_pre_tensor, Y_val_pre_tensor)
    
        start = timer()
    
        model_params = {'input_dim': input_dim,
                    'hidden_dim' : hidden_dim,
                    'layer_dim' : layer_dim,
                    'output_dim' : output_dim,
                    'dropout_prob' : dropout}

        model = get_model('lstm', model_params)
        model = model.to(device)

        batch_size = 32
        n_epochs = 1500

        learning_rate = 1e-5
        loss_fn = nn.MSELoss(reduction="mean")
        #loss_fn = MeanCubeLoss()

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
        Ftrain_loader = DataLoader(train_pre_eta, batch_size=batch_size, shuffle=False, drop_last=True)
        Fval_loader = DataLoader(val_pre_eta, batch_size=batch_size, shuffle=False, drop_last=True)
    
        opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer, patience = 30)
        opt.train(Ftrain_loader, Fval_loader, batch_size=batch_size, n_epochs=n_epochs, mode=i, n_features=input_dim, output_dim = output_dim)
        opt.plot_losses(output_dim)
            
        end = timer()

        dur = (end-start)/60
        print(f'The total duration for the training is {dur} minutes')

        X_Test = np.asarray(data_test[j,data_pre_vals.shape[1]-input_len:data_pre_vals.shape[1]])
        Y_Test = np.asarray(data_test[j,data_pre_vals.shape[1]:data_pre_vals.shape[1]+missing_len])

        Test_features = torch.Tensor(X_Test)
        Test_targets = torch.Tensor(Y_Test)

        #model = get_model('lstm', model_params)
        #model = model.to(device)

        #PATH = f'lstm_1.pt'
        #model.load_state_dict(torch.load(PATH))
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
        bl1 = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer, patience = 50)
        values, preds = bl1.evaluate2(Test_features,Test_targets, input_len, output_len)
    
        num = len(Test_targets) % output_len

        if (num != 0):
            preds = preds[:len(Test_targets)]
 
        p = np.asarray(preds).reshape(missing_len)
        Predictions_pre[j,:] = p
        Values[j,:] = values

    Preds_rescaled = scaler.inverse_transform(Predictions_pre.reshape(-1,1)).reshape(Predictions_pre.shape[0],Predictions_pre.shape[1])
    Vals_rescaled = scaler.inverse_transform(Values.reshape(-1,1)).reshape(Predictions_pre.shape[0],Predictions_pre.shape[1])
    
    np.savetxt(f"Preds_lstm_{i}.out", Preds_rescaled)
    np.savetxt(f"Vals_{i}.out", Vals_rescaled)
