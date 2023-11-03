import numpy as np
from neuralnet.Layer import Layer
from neuralnet.Activation import *
from neuralnet.clip_gradients import clip_gradients


class LSTM(Layer):
    def __init__(self, num_units: int, input_units: int):
        super().__init__()
        self.type = "lstm"
        self.num_units = num_units
        self.input_units = input_units
        self.feature_map_shape = num_units

        mean = 0
        std = 1

        # lstm cell weights
        self.forget_gate_weights = np.random.normal(
            mean, std, (self.input_units + self.num_units, self.num_units)
        )
        self.input_gate_weights = np.random.normal(
            mean, std, (self.input_units + self.num_units, self.num_units)
        )
        self.output_gate_weights = np.random.normal(
            mean, std, (self.input_units + self.num_units, self.num_units)
        )
        self.cell_gate_weights = np.random.normal(
            mean, std, (self.input_units + self.num_units, self.num_units)
        )

        self.output_size = (1, self.num_units)
        self.output = None

    def lstm_cell(self, data, prev_activation_matrix, prev_cell_matrix):
        # get parameters
        fgw = self.forget_gate_weights
        igw = self.input_gate_weights
        ogw = self.output_gate_weights
        cgw = self.cell_gate_weights

        # concat batch data and prev_activation matrix
        concat_data = np.concatenate((data, prev_activation_matrix), axis=1)

        # forget gate activations
        fa = np.matmul(concat_data, fgw)
        fa = Sigmoid().calculate(fa)

        # input gate activations
        ia = np.matmul(concat_data, igw)
        ia = Sigmoid().calculate(ia)

        # output gate activations
        oa = np.matmul(concat_data, ogw)
        oa = Sigmoid().calculate(oa)

        # gate gate activations
        ca = np.matmul(concat_data, cgw)
        ca = Tanh().calculate(ca)

        # new cell memory matrix
        cell_memory_matrix = np.multiply(fa, prev_cell_matrix) + np.multiply(ia, ca)

        # current activation matrix
        activation_matrix = np.multiply(oa, Tanh().calculate(cell_memory_matrix))

        return cell_memory_matrix, activation_matrix

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
        timestep = input.shape[0]
        n_dim = input.shape[1]

        # initial activation_matrix(a0) and cell_matrix(c0)
        a0 = np.zeros([1, self.num_units], dtype=np.float32)
        c0 = np.zeros([1, self.num_units], dtype=np.float32)

        # unroll the names
        for i in range(len(input)):
            # get first first character batch
            data = input[i].reshape(1, n_dim)

            # lstm cell
            ct, at = self.lstm_cell(data, a0, c0)

            # update a0 and c0 to new 'at' and 'ct' for next lstm cell
            a0 = at
            c0 = ct

        self.output = at
