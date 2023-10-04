from neuralnet import Layer
import numpy as np
import json

from neuralnet.Encoder import MyEncoder


class NN:
    def __init__(self, input_shape: tuple, layers:Layer=[]):
        self.input_shape = input_shape
        self.layers = layers

    def add(self, layer):
        layer_length = len(self.layers)
        if layer_length == 0:
            input_shape = self.input_shape
        else:
            input_shape = self.layers[layer_length - 1].feature_map_shape
        self.layers.append(layer)
        self.layers[layer_length].calculate_feature_map_shape(input_shape)

    def forward_propagate(self, image: np.ndarray):
        # image = preprocess_image(image)
        for layer in self.layers:
            image = layer.forward_propagate(image)
            # print(image.shape)
        return image
    
    # def __iter__(self):
    #     yield from {
    #         "input_shape": self.input_shape,
    #         "layers": self.layers.__repr__,
    #     }.items()

    # def __str__(self):
    #     return json.dumps(self, cls=MyEncoder, ensure_ascii=False)

    # def __repr__(self):
    #     return self.__str__()
    
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

    def save_model(self, filename, indent):
        ind = ""
        for i in range(indent):
            ind+= " "
        print(ind)
        print(self.input_shape.__class__)
        # save_string = json.dumps({"input_shape": self.input_shape, "layers": self.layers}, cls=MyEncoder, indent=indent, ensure_ascii=False)
        with open(filename, 'w') as f:
            f.write("{\n")
            f.write(f'''{ind}"input_shape": {list(self.input_shape)},\n''')
            f.write(f'''{ind}"layers": [\n''')
            for i in range(len(self.layers)):
                f.write(f'''{ind}{ind}{self.layers[i]}''')
                if(i== len(self.layers)-1):
                    f.write('\n')
                else:
                    f.write(',\n')
            f.write(f'''{ind}]\n''')
            f.write("}\n")

    # def load_model(self, filename):
    #     # Opening JSON file
    #     f = open('filename', "r")
        
    #     data = json.load(f)