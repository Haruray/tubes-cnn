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
    
    def back_propagate(self, input, label, learning_rate):
        layers_count = len(self.layers)
        if(layers_count <= 0):
            raise Exception("There is no layers to backpropagate")
        prev_layer = self.layers[layers_count-1]
        last_deriv = prev_layer.detector_function.deriv(prev_layer.last_input, label)
        for i in range(layers_count-1, 0, -1):
            print("yeah")
    
    
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

    def save_model(self, filename, indent):
        ind = ""
        for i in range(indent):
            ind+= " "
        print(ind)
        print(self.input_shape.__class__)
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