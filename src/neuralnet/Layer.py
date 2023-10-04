import numpy as np


class Layer:
    def __init__(self):
        self.type = ""

    def forward_propagate(self, input: np.ndarray):
        pass

    def backpropagate(self, out: np.ndarray, learn_rate: float):
        pass

    def calculate_feature_map_shape(self, input: tuple):
        pass

    def __iter__(self):
        pass

    def __str__(self):
        pass

    def __repr__(self):
        pass