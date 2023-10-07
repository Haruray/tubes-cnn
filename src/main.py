import cv2
from neuralnet import NN, ConvLayer
from neuralnet import Pooling
from neuralnet import Flatten
from neuralnet import Dense
import numpy as np
import json

from neuralnet.Encoder import MyEncoder
from neuralnet.Trainer import Trainer

image = cv2.imread("251.jpeg")

def load_model(filename):
    # Opening JSON file
    f = open(filename, "r")
    
    # returns JSON object as
    # a dictionary
    data = json.load(f)
    if(data['input_shape']!=None):
        newModel = NN(data['input_shape'])
        layers = data['layers']
        if(layers!=None):
            for layer in layers:
                layers_count = len(newModel.layers)
                layer_type = layer['type']
                newLayer = None
                #Case Convolution layer
                if(layer_type == "conv2d"):
                    input_shape = tuple(layer['input_shape'])
                    num_filters = int(layer['num_filters'])
                    filter_size = tuple(layer['filter_size'])
                    stride = int(layer['stride'])
                    detector_function = str(layer['detector_function']['name'])
                    padding = int(layer['padding'])
                    newLayer = ConvLayer(input_shape, num_filters, filter_size, stride, detector_function, padding)
                    filter = np.array(layer['filter'])
                    if(layers_count > 0 and input_shape != newModel.layers[layers_count-1].feature_map_shape):
                        raise Exception(f'input shape:{input_shape} does not compatible for the previous output shape:{newModel.layers[layers_count-1].feature_map_shape}')
                    if(filter.shape == newLayer.filter.shape):
                        newLayer.filter = filter

                #Case Pooling layer                
                elif(layer_type == "pooling"):
                    mode = str(layer['mode'])
                    pool_size = tuple(layer['pool_size'])
                    stride = int(layer['stride'])
                    newLayer = Pooling(mode, pool_size, stride)

                #Case Dense Layer
                elif(layer_type == "dense"):
                    num_units = int(layer['num_units'])
                    detector_function = str(layer['detector_function']['name'])
                    input_len = None
                    if(layers_count == 0):
                        input_len = newModel.input_shape
                    else:
                        input_len = newModel.layers[layers_count-1].feature_map_shape
                    if(input_len != len(layer['weights'][0])):
                        raise Exception(f"The paramater size dos not match with previous output shape {input_len}")
                    if(input_len != None):
                        newLayer = Dense(num_units, input_len, detector_function)
                        newLayer.weights = np.array(layer['weights'])
                        newLayer.biases = np.array(layer['biases'])
                
                #Case Flatten Layer
                elif(layer_type == "flatten"):
                    newLayer = Flatten()

                else:
                    raise Exception(f'Type {layer_type} not recognized, try conv2d, pooling, dense, or flatten')
                
                newModel.add(newLayer)
        return newModel
    else:
        raise Exception('No input shape detected')


# Build the model
newModel = load_model("yeah2.json")

result = newModel.forward_propagate(image)

if result <= 0.5:
    category = "Bear"
else:
    category = "Panda"

# print(category)

# print(newModel.layers[5].weights.shape)

# print(newModel.layers[4].weights)
trainer = Trainer(newModel, 1, 0.1, np.array([image]), np.array([[0]]))
trainer.fit()
# print(newModel.layers[4].weights)


# model = NN(image.shape)
# model.add(
#     ConvLayer(
#         input_shape=image.shape,
#         padding=0,
#         num_filters=1,
#         filter_size=(3, 3),
#         stride=2,
#         detector_function="relu",
#     )
# )
# model.add(Pooling(mode="max", pool_size=(2, 2), stride=2))
# model.add(Flatten())

# flat_shape = model.layers[2].feature_map_shape
# model.add(Dense(1024, flat_shape, "relu"))

# model.add(Dense(128, 1024, "relu"))

# model.add(Dense(64, 128, "relu"))

# flat_shape = model.layers[4].feature_map_shape
# model.add(Dense(1, 64, "sigmoid"))
# model.save_model("yeah2.json", 4)