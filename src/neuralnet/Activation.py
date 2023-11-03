import numpy as np
import json

from neuralnet.Encoder import MyEncoder


class Activation:
    def __init__(self):
        self.name = ""

    def calculate(self, input: np.ndarray):
        pass

    def deriv(self, input: np.ndarray, pred: np.ndarray = None):
        pass

    def __iter__(self):
        yield from {"name": self.name}.items()

    def __str__(self):
        return json.dumps(dict(self), cls=MyEncoder, ensure_ascii=False)

    def __repr__(self):
        return self.__str__()


class Relu(Activation):
    def __init__(self):
        super().__init__()
        self.name = "relu"

    def calculate(self, input: np.ndarray):
        return np.maximum(input, np.zeros(input.shape))

    def deriv(self, input: np.ndarray, pred: np.ndarray = None):
        output = input.copy()
        output[output < 0] = 0
        output[output >= 0] = 1
        return output


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()
        self.name = "sigmoid"

    def calculate(self, input: np.ndarray):
        return 1 / (1 + np.exp(-input))

    def deriv(self, input: np.ndarray, pred: np.ndarray = None):
        # print(input)
        # print(pred)
        # print(self.calculate(input))
        # print(self.calculate(input) * (1 - self.calculate(input)))
        return self.calculate(input) * (1 - self.calculate(input))


class Softmax(Activation):
    def __init__(self):
        super().__init__()
        self.name = "softmax"

    def calculate(self, input: np.ndarray):
        return np.exp(input) / np.sum(np.exp(input))

    def deriv(self, input: np.ndarray, pred: np.ndarray = None):
        copy = np.copy(pred)
        copy[copy == input] = -(1 - copy[copy == input])
        return copy


class Tanh(Activation):
    def __init__(self):
        super().__init__()
        self.name = "tanh"

    def calculate(self, input: np.ndarray):
        return np.tanh(input)
