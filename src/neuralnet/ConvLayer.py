from neuralnet.Layer import Layer
from neuralnet.Activation import *
import numpy as np


class ConvLayer(Layer):
    def __init__(
        self,
        input_shape: tuple,
        padding: int,
        num_filters: int,
        filter_size: tuple,
        stride: int,
        detector_function: str,
    ):
        super().__init__()
        self.type = "conv2d"
        # defined parameters
        self.input_shape = input_shape
        self.padding = padding
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        # check if input_shape and filter_size valid
        if not self.check_image_filter_validity():
            raise Exception("Error : Number of image channel and filter don't match")
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
        self.filter = (
            np.random.randn(self.num_filters, self.filter_size[0], self.filter_size[1])
            * 0.1
        )

    def get_image_channels(self, matrix: np.ndarray):
        channels = []
        if len(matrix.shape) == 3:
            for i in range(matrix.shape[2]):
                channels.append(matrix[:, :, i])
        else:
            return [matrix]  # berarti dia single channel
        return channels

    def check_image_filter_validity(self):
        # check if the image matrix have multi channel (rgb) or not, and match it with the filter size
        if len(self.input_shape) == 3:
            return self.input_shape[2] == self.filter_size[1]
        else:
            return True

    def extract_features(self, matrix: np.ndarray):
        height = matrix.shape[0]
        width = matrix.shape[1]
        # center of filter matrix
        center = (self.filter_size[0] // 2, self.filter_size[1] // 2)
        i = j = 0
        # loop matrix gambarnya buat ekstraksi fitur
        while i < height:
            j = 0
            while j < width:
                if i + self.filter_size[0] < height and j + self.filter_size[1] < width:
                    region = matrix[
                        i : (i + self.filter_size[0]), j : (j + self.filter_size[1])
                    ]
                    # bagian (region) dari matrix yang sudah di ekstrak , koordinat x pada feature map, koordinat y pada feature map
                    idx_i = max(
                        0, i if self.padding == 0 else i + center[0] - self.padding
                    )
                    idx_j = max(
                        0, j if self.padding == 0 else j + center[1] - self.padding
                    )
                    yield region, idx_i, idx_j
                j += self.stride
            i += self.stride

    def detector(self, matrix: np.ndarray):
        return self.detector_function.calculate(matrix)

    def forward_propagate(self, input: np.ndarray):
        # input adalah matrix gambar
        # kita tambahkan padding pada matrix gambar
        og_height = input.shape[0]
        og_width = input.shape[1]
        if len(input.shape) == 3:
            input.resize(
                (og_height + self.padding, og_width + self.padding, input.shape[2]),
                refcheck=False,
            )
        else:
            input.resize(
                (og_height + self.padding, og_width + self.padding), refcheck=False
            )
        # modified height and width
        height = input.shape[0]
        # berdasarkan rumus di ppt...i think
        feature_map_v = (
            int((height - self.filter_size[0] + 2 * self.padding) / self.stride) + 1
        )
        feature_map = np.zeros((feature_map_v, feature_map_v, self.num_filters))
        input_channels = self.get_image_channels(input)
        for channel in input_channels:
            for region, i, j in self.extract_features(channel):
                # bagian (region) yang sudah di ekstrak di kalikan dengan filter yang ada. Argumen "axis" aku belum tau buat apa..
                feature_map[i, j] += np.sum(region * self.filter)

        feature_map = self.detector(feature_map)

        return feature_map
