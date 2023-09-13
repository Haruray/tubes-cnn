import numpy as np
from neuralnet.Layer import Layer
from neuralnet.Activation import *

class Dense(Layer):
    def __init__(self, num_units: int, input_len: int, detector_function: str):
        super().__init__()

        self.type = "dense"
        self.num_units = num_units

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
        self.weights = (np.random.randn(self.num_units, input_len)*0.1)

        # bias
        self.biases = (np.zeros(num_units))

    def forward_propagate(self, input: np.ndarray):
        input = input.flatten() # flatten input
        output = np.dot(input, self.weights.T) + self.biases # forward propagate
        output = self.detector_function.calculate(output) # activation 
        return output