from neuralnet.Layer import Layer
import numpy as np


class ConvLayer(Layer):
    def __init__(
        self,
        input_shape: tuple,
        padding: int,
        num_filters: int,
        filter_size: tuple,
        stride: int,
    ):
        super().__init__()
        self.type = "conv2d"
        # defined parameters
        self.input_shape = input_shape
        self.padding = padding
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride

        # weight
        self.filter = np.random.randn(
            self.num_filters, self.filter_size[0], self.filter_size[1]
        )

    def iterate(self, matrix: np.ndarray):
        height = matrix.shape[0]
        width = matrix.shape[1]
        # center of filter matrix
        center = (self.filter_size[0] // 2, self.filter_size[1] // 2)
        i = j = 0
        # loop matrix gambarnya buat ekstraksi fitur
        while i < height:
            while j < width:
                if (
                    i + self.filter_size[0] <= height
                    and j + self.filter_size[1] <= width
                ):
                    region = matrix[
                        i : (i + self.filter_size[0]), j : (j + self.filter_size[1])
                    ]
                    # bagian (region) dari matrix yang sudah di ekstrak , koordinat x pada feature map, koordinat y pada feature map
                    idx_i = i if self.padding == 0 else i + center[0] - self.padding
                    idx_j = j if self.padding == 0 else j + center[1] - self.padding
                    yield region, idx_i, idx_j
                j += self.stride
            i += self.stride

    def forward_propagate(self, input: np.ndarray):
        # input adalah matrix gambar
        # kita tambahkan padding pada matrix gambar
        og_height = input.shape[0]
        og_width = input.shape[1]
        input.resize((og_height + self.padding, og_width + self.padding, 3), refcheck=False)
        # modified height and width
        height = input.shape[0]
        # berdasarkan rumus di ppt...i think
        feature_map_v = (
            int((height - self.filter_size[0] + 2 * self.padding) / self.stride) + 1
        )
        feature_map = np.zeros((feature_map_v, feature_map_v, self.num_filters))
        for region, i, j in self.iterate(input):
            # bagian (region) yang sudah di ekstrak di kalikan dengan filter yang ada. Argumen "axis" aku belum tau buat apa..
            feature_map[i, j] = np.sum(region * self.filter) # sepertinya masih salah

        return feature_map
