import numpy as np

class Activation:
    def __init__(self) :
        self.name = ''
    def calculate(self, input: np.ndarray):
        pass
    def deriv(self, input: np.ndarray):
        pass

class Relu(Activation):
    def __init__(self):
        super().__init__()
        self.name = 'relu'
    def calculate(self, input: np.ndarray):
        return np.maximum(input, np.zeros(input.shape))

class Sigmoid(Activation):
    def __init__(self):
        super().__init__()
        self.name = 'sigmoid'
    def calculate(self, input: np.ndarray):
        return (1/(1 + np.exp(-input)))

class Softmax(Activation):
    def __init__(self):
        super().__init__()
        self.name = 'softmax'
    def calculate(self, input: np.ndarray):
        return np.exp(input) / np.sum(np.exp(input))