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
    
    def train(self, train_loader, val_loader, batch_size, n_epochs, n_features, conv_layers):
        
        model_path = f'cnn_1D_with_{n_features}_as_training_length.pt'
        
        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                #x_batch = x_batch.view([batch_size, n_features]).to(device)
                x_batch = x_batch.unsqueeze(1).to(device)
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
                    x_val = x_val.unsqueeze(1).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch % 100 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
                )
                
                #plt.figure(figsize=[15,5])
                #plt.plot(y_val[-1].cpu(), color = 'black', linestyle = '-')
                #plt.plot(yhat[-1].cpu(), color = 'blue', linestyle = '-')
                #plt.legend(['Actual validation data', 'Network validation results'])
                #plt.title(f"Comparisons after {epoch} epochs")
                #plt.grid()
                #plt.savefig('Validation comparisons after %d epochs.png'%epoch,dpi=300)
                #plt.close()
                
        #model_path = 'cnn_1D_method_example1.pt'
        #model_path = f'dwt_method_for_{training_len}_as_training_length and {decomposition_level} as level and {hidden_dim} as hidden dim and {hidden_dim2} as hidden_dim2.pt'
        torch.save(self.model.state_dict(), model_path)
    
    def evaluate(self, x, test, training_len, missing_len):
        with torch.no_grad():
            predictions = []
            values = []
            for i in range(len(test)):
                x = x.to(device)
                self.model.eval()
                x_test = x.view([1, -1, training_len]).to(device)
                yhat = self.model(x_test)
                yint = torch.reshape(yhat,(missing_len,1,1))
                y_int = yint.to(device).cpu()
                predictions.append(y_int[0].detach().numpy())
                x=torch.reshape(x,(training_len,1))
                x = torch.cat((x,yint[0]),0)
                x = x[1:]
        preds =  torch.reshape(torch.Tensor(predictions),(len(predictions),1))
        
        return np.asarray(preds)
    
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
    
    def plot_losses(self,n_epochs,batch_size,learning_rate, training_len, conv_layers):
        """The method plots the calculated loss values for training and validation
        """
        np.savetxt(f"Training_length={training_len}_train.out", self.train_losses, fmt='%1.4e')
        np.savetxt(f"Training_length={training_len}_val.out", self.val_losses, fmt='%1.4e')
        
        plt.figure(figsize=[10,8])
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title(f"Losses for training length = {training_len}")
        plt.grid()
        plt.show()
        plt.savefig(f'Losses comparisons for training length={training_len} over epochs.png',dpi=300)
        plt.close()
        

input_dim = input_len
output_dim = output_len
layer_dim = 2
dropout = 0.3
weight_decay = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_fn = nn.MSELoss(reduction="mean")

learning_rate = 1e-5
batch_size = 1
n_epochs = 1000

X_Test = np.asarray(Test_Matrix_Features)
Y_Test = np.asarray(Test_Matrix_Targets)
val_int = scaler.inverse_transform(Y_Test.reshape(-1,1))
Values = val_int.reshape(len(val_int),1)
    
Test_features = torch.Tensor(X_Test)
Test_targets = torch.Tensor(Y_Test)

hidden_dim = 128

model_params = {'input_dim': input_dim,
                'hidden_dim' : hidden_dim,
                'layer_dim' : layer_dim,
                'output_dim' : output_dim,
                'dropout_prob' : dropout}

model = get_model('lstm', model_params)
model = model.to(device)

PATH = f"lstm_twoFrequencies.pt"
model.load_state_dict(torch.load(PATH))
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
bl1 = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
preds = bl1.evaluate2(Test_features,Test_targets, input_len, output_len)
num = len(Test_targets) % output_len

if (num != 0):
    preds = preds[:len(Test_targets)]

pred_int = scaler.inverse_transform(preds.reshape(-1,1))
Predictions = pred_int.reshape(len(pred_int))

np.savetxt("twoFreqs_lstm_pred_results.out", Predictions)
np.savetxt("twoFreqs_values.out", Values)