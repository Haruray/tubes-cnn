import numpy as np
from neuralnet.Layer import Layer
from neuralnet.Activation import *


class LSTM(Layer):
    """
    LSTM

    Args:
    input_units (int): Number of input units.
    num_units (int): Number of LSTM units.

    """

    def __init__(self, input_units: int, num_units: int):
        super().__init__()
        self.type = "lstm"
        self.num_units = num_units
        self.input_units = input_units
        self.feature_map_shape = num_units

        # cell weights
        self.forget_weights = np.random.randn(
            self.input_units + self.num_units, self.num_units
        )
        self.forget_biases = np.random.randn(1, self.num_units)

        self.input_weights = np.random.randn(
            self.input_units + self.num_units, self.num_units
        )
        self.input_biases = np.random.randn(1, self.num_units)

        self.output_weights = np.random.randn(
            self.input_units + self.num_units, self.num_units
        )
        self.output_biases = np.random.randn(1, self.num_units)

        self.cell_hat_weights = np.random.randn(
            self.input_units + self.num_units, self.num_units
        )
        self.cell_hat_biases = np.random.randn(1, self.num_units)

    def lstm_cell_calculations(self, data, prev_o, prev_cell):
        """
        Calculates the LSTM cell.

        """
        input_and_prev_o = np.concatenate((data, prev_o), axis=1)

        # forget
        fg = Sigmoid().calculate(
            np.matmul(input_and_prev_o, self.forget_weights) + self.forget_biases
        )
        # input
        ig = Sigmoid().calculate(
            np.matmul(input_and_prev_o, self.input_weights) + self.input_biases
        )
        # output
        og = Sigmoid().calculate(
            np.matmul(input_and_prev_o, self.output_weights) + self.output_biases
        )
        # cell hat
        ch = Tanh().calculate(
            np.matmul(input_and_prev_o, self.cell_hat_weights) + self.cell_hat_biases
        )
        # new cell
        cell = fg * prev_cell + ig * ch

        logits = og * Tanh().calculate(cell)

        return cell, logits

    def __iter__(self):
        yield from {
            "type": self.type,
            "num_units": self.num_units,
            "input_units": self.input_units,
            "forget_weights": self.forget_weights.tolist(),
            "input_weights": self.input_weights.tolist(),
            "output_weights": self.output_weights.tolist(),
            "cell_hat_weights": self.cell_hat_weights.tolist(),
            "forget_biases": self.forget_biases.tolist(),
            "input_biases": self.input_biases.tolist(),
            "output_biases": self.output_biases.tolist(),
            "cell_hat_biases": self.cell_hat_biases.tolist(),
        }.items()

    def __str__(self):
        return json.dumps(dict(self), cls=MyEncoder, ensure_ascii=False)

    def __repr__(self):
        return self.__str__()

    def forward_propagate(self, input: np.ndarray):
        """
        Forward propagates the input through the LSTM layer.

        """
        # print(input)

        n_feat = input.shape[1]

        # prev output and prev cell
        o_prev = np.zeros([1, self.num_units])
        c_prev = np.zeros([1, self.num_units])

        for i in range(len(input)):
            data = input[i].reshape(1, n_feat)
            ct, ot = self.lstm_cell_calculations(data, o_prev, c_prev)
            o_prev = ot
            c_prev = ct

        return ot

    def backpropagate(self, out: np.ndarray, learn_rate: float):
        """
        Backpropagates the output through the LSTM layer.
        pass, not implemented

        """
        return super().backpropagate(out, learn_rate)
