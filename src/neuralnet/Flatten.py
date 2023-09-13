import numpy as np
from neuralnet.Layer import Layer

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.type = 'flatten'
    
    def forward_propagate(self, input: np.ndarray):
        # Flatten input: membuat input menjadi array satu dimensi
        output = input.flatten()
        return output

