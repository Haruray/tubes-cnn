import Layer
class MaxPooling(Layer):
    def __init__(self):
        super().__init__()
        self.type = 'max_pooling2d'