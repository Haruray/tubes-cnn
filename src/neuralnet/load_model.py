from neuralnet.NN import NN
from neuralnet.ConvLayer import ConvLayer
from neuralnet.Pooling import Pooling
from neuralnet.Flatten import Flatten
from neuralnet.Dense import Dense
from neuralnet.lstm import LSTM
import numpy as np
import json


def load_model(filename):
    # Opening JSON file
    f = open(filename, "r")

    # returns JSON object as
    # a dictionary
    data = json.load(f)
    if data["input_shape"] != None:
        newModel = NN(data["input_shape"])
        layers = data["layers"]
        if layers != None:
            for layer in layers:
                layers_count = len(newModel.layers)
                layer_type = layer["type"]
                newLayer = None
                # Case Convolution layer
                if layer_type == "conv2d":
                    input_shape = tuple(layer["input_shape"])
                    num_filters = int(layer["num_filters"])
                    filter_size = tuple(layer["filter_size"])
                    stride = int(layer["stride"])
                    detector_function = str(layer["detector_function"]["name"])
                    padding = int(layer["padding"])
                    newLayer = ConvLayer(
                        input_shape,
                        num_filters,
                        filter_size,
                        stride,
                        detector_function,
                        padding,
                    )
                    filter = np.array(layer["filter"])
                    if (
                        layers_count > 0
                        and input_shape
                        != newModel.layers[layers_count - 1].feature_map_shape
                    ):
                        raise Exception(
                            f"input shape:{input_shape} does not compatible for the previous output shape:{newModel.layers[layers_count-1].feature_map_shape}"
                        )
                    if filter.shape == newLayer.filter.shape:
                        newLayer.filter = filter

                # Case Pooling layer
                elif layer_type == "pooling":
                    mode = str(layer["mode"])
                    pool_size = tuple(layer["pool_size"])
                    stride = int(layer["stride"])
                    newLayer = Pooling(mode, pool_size, stride)

                # Case Dense Layer
                elif layer_type == "dense":
                    num_units = int(layer["num_units"])
                    detector_function = str(layer["detector_function"]["name"])
                    input_len = None
                    if layers_count == 0:
                        input_len = newModel.input_shape
                    else:
                        input_len = newModel.layers[layers_count - 1].feature_map_shape
                    if input_len != len(layer["weights"][0]):
                        raise Exception(
                            f"The paramater size dos not match with previous output shape {input_len}"
                        )
                    if input_len != None:
                        newLayer = Dense(num_units, input_len, detector_function)
                        newLayer.weights = np.array(layer["weights"])
                        newLayer.biases = np.array(layer["biases"])

                # Case Flatten Layer
                elif layer_type == "flatten":
                    newLayer = Flatten()

                elif layer_type == "lstm":
                    num_units = int(layer["num_units"])
                    input_units = int(layer["input_units"])
                    newLayer = LSTM(input_units, num_units)
                    newLayer.forget_weights = np.array(layer["forget_weights"])
                    newLayer.forget_biases = np.array(layer["forget_biases"])
                    newLayer.input_weights = np.array(layer["input_weights"])
                    newLayer.input_biases = np.array(layer["input_biases"])
                    newLayer.output_weights = np.array(layer["output_weights"])
                    newLayer.output_biases = np.array(layer["output_biases"])
                    newLayer.cell_hat_weights = np.array(layer["cell_hat_weights"])
                    newLayer.cell_hat_biases = np.array(layer["cell_hat_biases"])

                else:
                    raise Exception(
                        f"Type {layer_type} not recognized, try conv2d, pooling, dense, or flatten"
                    )

                newModel.add(newLayer)
        return newModel
    else:
        raise Exception("No input shape detected")
