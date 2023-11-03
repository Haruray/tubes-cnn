import numpy as np
from neuralnet.Layer import Layer
from neuralnet.Activation import *


class LSTM(Layer):
    def __init__(self, input_units: int, num_units: int):
        super().__init__()
        self.type = "lstm"
        self.num_units = num_units
        self.input_units = input_units
        self.feature_map_shape = num_units

        # cell weights
        self.forget_gate_weights = np.random.randn(
            self.input_units + self.num_units, self.num_units
        )
        self.input_gate_weights = np.random.randn(
            self.input_units + self.num_units, self.num_units
        )
        self.output_gate_weights = np.random.randn(
            self.input_units + self.num_units, self.num_units
        )
        self.cell_gate_weights = np.random.randn(
            self.input_units + self.num_units, self.num_units
        )

    def lstm_cell(self, data, prev_activation_matrix, prev_cell_matrix):
        # cells
        fg = self.forget_gate_weights
        ig = self.input_gate_weights
        og = self.output_gate_weights
        cg = self.cell_gate_weights

        input_and_prev_o = np.concatenate((data, prev_activation_matrix), axis=1)

        # forget
        fa = Sigmoid().calculate(np.matmul(input_and_prev_o, fg))
        # input
        ia = Sigmoid().calculate(np.matmul(input_and_prev_o, ig))
        # output
        oa = Sigmoid().calculate(np.matmul(input_and_prev_o, og))
        # cell hat
        ca = Tanh().calculate(np.matmul(input_and_prev_o, cg))
        # new cell
        cell = np.multiply(fa, prev_cell_matrix) + np.multiply(ia, ca)

        logits = np.multiply(oa, Tanh().calculate(cell))

        return cell, logits

    def __iter__(self):
        yield from {
            "type": self.type,
            "num_units": self.num_units,
            "input_units": self.input_units,
            "forget_weights": self.forget_gate_weights.tolist(),
            "input_weights": self.input_gate_weights.tolist(),
            "output_weights": self.output_gate_weights.tolist(),
            "cell_weights": self.cell_gate_weights.tolist(),
        }.items()

    def __str__(self):
        return json.dumps(dict(self), cls=MyEncoder, ensure_ascii=False)

    def __repr__(self):
        return self.__str__()

    def forward_propagate(self, input: np.ndarray):
        n_dim = input.shape[1]

        # prev output and prev cell
        o_prev = np.zeros([1, self.num_units])
        c_prev = np.zeros([1, self.num_units])

        for i in range(len(input)):
            data = input[i].reshape(1, n_dim)

            ct, ot = self.lstm_cell(data, o_prev, c_prev)

            o_prev = ot
            c_prev = ct

        return ot
