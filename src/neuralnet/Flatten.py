import Layer
class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.type = 'flatten'