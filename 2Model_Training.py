#Code to train the model to forecast the data for a selected country

import torch
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim


#Initialization
Location = 'Italy' #Select the country for training

#Importing data from the updated file
data = pd.read_csv('corona.csv',index_col=0, parse_dates=[0])
country = data.loc[data['location'] == Location, 'location':'new_deaths_per_million']
country.set_index('date', inplace = True)
country = country[(country[['total_cases']] != 0).all(axis=1)] #we will focus only on the dates after the first case detected
values = country.values

groups = [1, 2, 3, 4, 5, 6, 7]
i = 1
#Plot each statistic and save the charts
plt.figure()
for group in groups:
	plt.subplot(len(groups), 1, i)
	plt.plot(values[:, group])
	plt.title(country.columns[group], y=0.5, loc='left')
	i += 1
plt.savefig('plots/Stats-' + Location + '.png')
    
#Select the most important statistics looking at the graph - New cases and New deaths
#convert the dataframe
daily_cases = country.iloc[:,2]
daily_cases.index = pd.to_datetime(daily_cases.index)


#In the first phase, we will use 20% available for testing and the remaining part for training
test_sample = int(len(daily_cases)*0.2)

training_part = daily_cases[:-test_sample]
testing_part = daily_cases[-test_sample:]

training_part.shape

Scaler = MinMaxScaler()

Scaler = Scaler.fit(np.expand_dims(training_part, axis=1))

training_part = Scaler.transform(np.expand_dims(training_part, axis=1))

testing_part = Scaler.transform(np.expand_dims(testing_part, axis=1))


def create_dataset(input_set, input_size):
    list_x, list_y = [], []

    for i in range(len(input_set)-input_size-1):
        x = input_set[i:(i+input_size)]
        y = input_set[i+input_size]
        list_x.append(x)
        list_y.append(y)
    return np.array(list_x), np.array(list_y)

#We will train the model with data sequences of 9 days (not much data available)
input_size = 9
training_x, training_y = create_dataset(training_part, input_size)
testing_x, testing_y = create_dataset(testing_part, input_size)

training_x = torch.from_numpy(training_x).float()
training_y = torch.from_numpy(training_y).float()

testing_x = torch.from_numpy(testing_x).float()
testing_y = torch.from_numpy(testing_y).float()

print(testing_x)
print(testing_y)

#Creating a prediction model
class Predict_Cases(nn.Module):

  def __init__(self, n_features, hidden_dim, seq_len, num_layers=2):
    super(Predict_Cases, self).__init__()

    self.hidden_dim = hidden_dim
    self.seq_len = seq_len
    self.num_layers = num_layers

#Long short-term memory well suited to predict the time-series data
    self.lstm = nn.LSTM(
      input_size=n_features,
      hidden_size=hidden_dim,
      num_layers=num_layers,
      dropout = 0.6
    )

    self.linear = nn.Linear(in_features=hidden_dim, out_features=1)

#Reset the hidden dimension
  def restart_hid_dim(self):
    self.hidden = (
        torch.zeros(self.num_layers, self.seq_len, self.hidden_dim),
        torch.zeros(self.num_layers, self.seq_len, self.hidden_dim)
    )

  def forward(self, sequences):
    lstm_out, self.hidden = self.lstm(
      sequences.view(len(sequences), self.seq_len, -1),
      self.hidden
    )
    last_time_step = \
      lstm_out.view(self.seq_len, len(sequences), self.hidden_dim)[-1]
    prediction_y = self.linear(last_time_step)
    return prediction_y

#Method to train the model
def model_training(model, training_part, train_labels, testing_part=None, test_labels=None):
  loss_fn = torch.nn.MSELoss(reduction='sum')
  optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)
  num_epochs = 30

  training_history = np.zeros(num_epochs)
  testing_history = np.zeros(num_epochs)

  for t in range(num_epochs):
    model.restart_hid_dim()

    prediction_y = model(training_x)

    loss = loss_fn(prediction_y.float(), training_y)

    if testing_part is not None:
      with torch.no_grad():
        testing_prediction_y = model(testing_x)
        test_loss = loss_fn(testing_prediction_y.float(), testing_y)
      testing_history[t] = test_loss.item()

    training_history[t] = loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
  return model.eval(), training_history, testing_history

model = Predict_Cases(1, 512, seq_len=input_size, num_layers=2)
model, training_history, testing_history = model_training(model, training_x, training_y, testing_x, testing_y)

