from neuralnet import Layer
import numpy as np


class NN:
    def __init__(self, input_shape: tuple, layers=[]):
        self.input_shape = input_shape
        self.layers = layers

    def add(self, layer):
        layer_length = len(self.layers)
        if layer_length == 0:
            input_shape = self.input_shape
        else:
            input_shape = self.layers[layer_length - 1].feature_map_shape
        self.layers.append(layer)
        self.layers[layer_length].calculate_feature_map_shape(input_shape)

    def forward_propagate(self, image: np.ndarray):
        # image = preprocess_image(image)
        for layer in self.layers:
            image = layer.forward_propagate(image)
            # print(image.shape)
        return image
