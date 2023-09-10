from neuralnet.Layer import Layer


class Pooling(Layer):
    def __init__(self):
        super().__init__()
        self.type = "pooling"
