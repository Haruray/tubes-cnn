import cv2
from neuralnet import NN, ConvLayer
from neuralnet import Pooling
from neuralnet import Flatten
from neuralnet import Dense
from neuralnet import LSTM
from neuralnet import load_model
import numpy as np
from neuralnet import Trainer
from neuralnet import Preprocess
import pandas as pd

df = pd.read_csv("../data/lstm/Train_stock_market.csv")

data = np.array(df[["Low", "Open", "Volume", "High", "Close", "Adjusted Close"]])


def create_sequences(data, seq_length):
    sequences = []
    targets = []
    data_len = len(data)
    for i in range(data_len - seq_length):
        seq_end = i + seq_length
        seq_x = data[i:seq_end]
        seq_y = data[seq_end]
        sequences.append(seq_x)
        targets.append(seq_y)
    return np.array(sequences), np.array(targets)


timestep = 10
lstm_cells = 15
X_train, y_train = create_sequences(data, timestep)

model = NN(X_train.shape)
model.add(LSTM(X_train.shape[2], lstm_cells))
flat_shape = model.layers[0].feature_map_shape
model.add(Dense(6, flat_shape, "relu"))
model.save_model("lstm.json", 4)
print(model.forward_propagate(X_train[0]))
# print(X_train[0])
