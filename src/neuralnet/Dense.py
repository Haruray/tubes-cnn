import numpy as np
from neuralnet.Layer import Layer
from neuralnet.Activation import *


class Dense(Layer):
    def __init__(self, num_units: int, input_len: int, detector_function: str):
        super().__init__()

        self.type = "dense"
        self.num_units = num_units
        self.feature_map_shape = num_units

        # detector function
        if detector_function == "relu":
            self.detector_function = Relu()
        elif detector_function == "softmax":
            self.detector_function = Softmax()
        elif detector_function == "sigmoid":
            self.detector_function = Sigmoid()
        else:
            raise Exception("Activation function not recognized.")

        # weight
        self.weights = np.random.randn(self.num_units, input_len)

        # bias
        self.biases = np.zeros(num_units)
        
    def __iter__(self):
        yield from {
            "type": self.type,
            "num_units": self.num_units,
            "detector_function": self.detector_function,
            "weights": self.weights.tolist(),
            "biases": self.biases.tolist(),
        }.items()

    def __str__(self):
        return json.dumps(dict(self), cls=MyEncoder, ensure_ascii=False)

    def __repr__(self):
        return self.__str__()

    def forward_propagate(self, input: np.ndarray):
        input = input.flatten()  # flatten input
        output = np.dot(input, self.weights.T) + self.biases  # forward propagate
        output = self.detector_function.calculate(output)  # activation
        return output

    def calculate_feature_map_shape(self, input: tuple):
        return self.feature_map_shape
