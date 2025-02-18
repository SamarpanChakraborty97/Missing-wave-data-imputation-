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


x = np.arange(0,12*math.pi, math.pi/600)


first_freq = 0.4 * np.sin(20*x)
second_freq = 0.4 * np.sin(10*x)
third_freq = 0.4 * np.sin(3*x)
fourth_freq = 0.4 * np.sin(x)
fifth_ freq = 0.4 * np.sin(x/2)
sixth_freq = 0.4 * np.sin(x/5)


plt.figure(figsize=[15,5])
#plt.plot(x, low_freq, 'b', linewidth = 0.6)
#plt.plot(x, high_freq, 'r', linewidth = 0.6)
plt.plot(x, first_freq+second_freq+second_freq+third_freq+fourth_freq+fifth_freq+sixth_freq, 'k', linewidth = 0.6)

data = np.asarray(first_freq + second_freq + third_freq + fourth_freq + fifth_freq + sixth_freq)

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
# In[44]:


Eta_Df = pd.DataFrame([])

df = pd.DataFrame(train_val_data, columns = ['Eta'])
Eta_Df = Eta_Df.append(df, ignore_index = True)

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

input_len = 500
output_len = 1
Train_X, Train_Y, Val_X, Val_Y = generateLaggedDf(Eta_Df, input_len, output_len,'Eta')

X_Train = np.asarray(Train_X.iloc[:,::-1])
Y_Train = np.asarray(Train_Y.iloc[:,::-1])
X_Val = np.asarray(Val_X.iloc[:,::-1])
Y_Val = np.asarray(Val_Y.iloc[:,::-1])

torch.manual_seed(2)

Train_Eta_features = torch.Tensor(X_Train.copy())
Train_Eta_targets = torch.Tensor(Y_Train.copy())
Val_Eta_features = torch.Tensor(X_Val.copy())
Val_Eta_targets = torch.Tensor(Y_Val.copy())

train_Eta = TensorDataset(Train_Eta_features, Train_Eta_targets)
val_Eta = TensorDataset(Val_Eta_features, Val_Eta_targets)

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
        
    def train(self, train_loader, val_loader, batch_size, n_epochs, n_features, output_dim):

        model_path = f'lstm_twoFrequencies.pt'
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

start = timer()
input_dim = input_len
output_dim = output_len
hidden_dim = 128
layer_dim = 2
dropout = 0.3
weight_decay = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_params = {'input_dim': input_dim,
                'hidden_dim' : hidden_dim,
                'layer_dim' : layer_dim,
                'output_dim' : output_dim,
                'dropout_prob' : dropout}

model = get_model('lstm', model_params)
model = model.to(device)

loss_fn = nn.MSELoss(reduction="mean")

learning_rate = 1e-5
batch_size = 32
n_epochs = 1000

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

Ftrain_loader_eta = DataLoader(train_Eta, batch_size=batch_size, shuffle=False, drop_last=True)
Fval_loader_eta = DataLoader(val_Eta, batch_size=batch_size, shuffle=False, drop_last=True)

opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer, patience = 20)
opt.train(Ftrain_loader_eta, Fval_loader_eta, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim, output_dim = output_dim)
opt.plot_losses(output_dim)
            
end = timer()

dur = (end-start)/60
print(f'The total duration for the training is {dur} minutes')