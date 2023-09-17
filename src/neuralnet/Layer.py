import numpy as np


class Layer:
    def __init__(self):
        self.type = ""

    def forward_propagate(self, input: np.ndarray):
        pass

    def backpropagate(self, out: np.ndarray):
        pass
    
    def calculate_feature_map_shape(self, input:tuple):
        pass
