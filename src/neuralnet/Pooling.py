from neuralnet.Layer import Layer
import numpy as np


class Pooling(Layer):
    # intinya sama kaya ConvLayer, yaitu pada proses ekstraksinya. Yang berbeda adalah pada pemrosesannya, yaitu max/avg pooling
    def __init__(self, mode: str, pool_size: tuple, stride: int):
        super().__init__()
        self.type = "pooling"
        self.mode = mode
        self.pool_size = pool_size
        self.stride = stride

    def get_image_channels(self, matrix: np.ndarray):
        channels = []
        if len(matrix.shape) == 3:
            for i in range(matrix.shape[2]):
                channels.append(matrix[:, :, i])
        else:
            return [matrix]  # berarti dia single channel
        return channels

    def iterate(self, matrix: np.ndarray):
        height = matrix.shape[0]
        width = matrix.shape[1]
        new_height = height // self.pool_size[0]
        new_width = width // self.pool_size[1]
        i = j = 0
        # loop matrix gambarnya buat ekstraksi fitur
        while i < new_height:
            j = 0
            while j < new_width:
                if (
                    i + self.pool_size[0] < new_height
                    and j + self.pool_size[1] < new_width
                ):
                    region = matrix[
                        i
                        * self.pool_size[0] : (
                            i * self.pool_size[0] + self.pool_size[0]
                        ),
                        j
                        * self.pool_size[1] : (
                            j * self.pool_size[1] + self.pool_size[1]
                        ),
                    ]
                    # bagian (region) dari matrix yang sudah di ekstrak , koordinat x pada feature map, koordinat y pada feature map
                    yield region, i, j
                j += self.stride
            i += self.stride

    def forward_propagate(self, input: np.ndarray):
        # input adalah feature map hasil conv layer
        height = input.shape[0]
        num_filters = input.shape[2]
        # sama kaya rumus ukuran feature map kaya ConvLayer, cuma gapakai padding
        feature_map_v = int((height - self.pool_size[0]) / self.stride) + 1
        feature_map = np.zeros((feature_map_v, feature_map_v, num_filters))
        input_channels = self.get_image_channels(input)
        for channel in input_channels:
            for region, i, j in self.iterate(channel):
                # bagian (region) yang sudah di ekstrak di kalikan dengan filter yang ada. Argumen "axis" aku belum tau buat apa..
                if self.mode == "max":
                    feature_map[i, j] = np.amax(region)
                elif self.mode == "avg":
                    feature_map[i, j] = np.average(region)
        return feature_map
