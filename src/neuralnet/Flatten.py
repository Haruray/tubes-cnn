import json
import numpy as np
from neuralnet.Encoder import MyEncoder
from neuralnet.Layer import Layer


class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.type = "flatten"
        self.feature_map_shape = None

    def __iter__(self):
        yield from {
            "type": self.type,
        }.items()

    def __str__(self):
        return json.dumps(dict(self), cls=MyEncoder, ensure_ascii=False)

    def __repr__(self):
        return self.__str__()

    def forward_propagate(self, input: np.ndarray):
        # Flatten input: membuat input menjadi array satu dimensi
        output = input.flatten()
        return output

    def calculate_feature_map_shape(self, input: tuple):
        flat_dim = 1
        for dim in input:
            flat_dim *= dim
        self.feature_map_shape = flat_dim
        return self.feature_map_shape
