#predicting

import torch
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.dates import DateFormatter
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim

with torch.no_grad():
  test_seq = testing_x[:1]
  preds = []
  for _ in range(len(testing_x)):
    testing_prediction_y = model(test_seq)
    pred = torch.flatten(testing_prediction_y).item()
    preds.append(pred)
    new_seq = test_seq.numpy().flatten()
    new_seq = np.append(new_seq, [pred])
    new_seq = new_seq[1:]
    test_seq = torch.as_tensor(new_seq).view(1, input_size, 1).float()
    
true_cases = Scaler.inverse_transform(np.expand_dims(testing_y.flatten().numpy(), axis=0)).flatten()

predicted_cases = Scaler.inverse_transform(np.expand_dims(preds, axis=0)).flatten()

Scaler = MinMaxScaler()

Scaler = Scaler.fit(np.expand_dims(daily_cases, axis=1))

all_data = Scaler.transform(np.expand_dims(daily_cases, axis=1))

X_all, y_all = create_dataset(all_data, input_size)

X_all = torch.from_numpy(X_all).float()
y_all = torch.from_numpy(y_all).float()

model, training_history, _ = model_training(model, X_all, y_all)

DAYS_TO_PREDICT = 14

with torch.no_grad():
  test_seq = X_all[:1]
  preds = []
  for _ in range(DAYS_TO_PREDICT):
    testing_prediction_y = model(test_seq)
    pred = torch.flatten(testing_prediction_y).item()
    preds.append(pred)
    new_seq = test_seq.numpy().flatten()
    new_seq = np.append(new_seq, [pred])
    new_seq = new_seq[1:]
    test_seq = torch.as_tensor(new_seq).view(1, input_size, 1).float()
    
predicted_cases = Scaler.inverse_transform(
  np.expand_dims(preds, axis=0)
).flatten()


predicted_index = pd.date_range(
  start=daily_cases.index[-1],
  periods=DAYS_TO_PREDICT + 1,
  closed='right'
)

predicted_cases = pd.Series(
  data=predicted_cases,
  index=predicted_index
)

fig1, ax1 = plt.subplots()
ax1.plot(predicted_cases, label='Predicted Daily Cases')
myFmt = DateFormatter("%d.%m")
ax1.xaxis.set_major_formatter(myFmt)
fig1.autofmt_xdate()
_ = plt.xticks(rotation=90)
plt.savefig('plots/Predicted cases-' + Location + '.png')
plt.show()


fig2, ax2 = plt.subplots()
ax2.plot(daily_cases, label='Historical Daily Cases')
ax2.plot(predicted_cases, label='Predicted Daily Cases')
myFmt = DateFormatter("%d.%m")
ax2.xaxis.set_major_formatter(myFmt)
fig2.autofmt_xdate()
_ = plt.xticks(rotation=90)
plt.savefig('plots/Forecast-' + Location + '.png')
plt.show()

