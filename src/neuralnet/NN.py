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
    

    def calculate_derr_error(self, layer: Layer, preds: np.ndarray, labels: np.ndarray):
        if(len(preds) != len(labels)):
            raise Exception(f"The label: {len(labels)} and prediction: {len(preds)} does not have same length")
        # print("shape")
        # print(np.sum(labels - preds).shape)
        # print(layer.detector_function.deriv(layer.last_input, preds).shape)
        # print(layer.last_input.shape)
        return preds - labels
        
            
    
    def backpropagate(self, input, label, learning_rate):
        layers_count = len(self.layers)
        if(layers_count <= 0):
            raise Exception("There is no layers to backpropagate")
        result = self.forward_propagate(input)
        prev_layer = self.layers[layers_count-1]
        last_deriv = None
        # print("yeah")
        # print(last_deriv.shape)
        # print(last_deriv)
        for i in range(layers_count-1, 0, -1):
            layer = self.layers[i]
            
            #CASE OUTPUT LAYER (OBVIOUSLY DENSE LAYER)
            if(i == layers_count-1):
                last_deriv = self.calculate_derr_error(prev_layer, result, label)
                layer.backpropagate(last_deriv, learning_rate)

            # CASE HIDDEN LAYER
            else:
                #CASE TYPE DENSE
                if(layer.type == "dense"):
                    print(layer.detector_function.name)
                    print(layer.detector_function.deriv(layer.last_input).shape) #drelu2/dx3
                    print(prev_layer.weights.shape) #dnet/drelu2
                    print(last_deriv.shape) #deD/dnet
                    print(layer.last_input.shape) #dx3/dw2

                    print(layer.weights.shape)
                    last_deriv = layer.detector_function.deriv(layer.last_input * layer.weights) * prev_layer.weights * last_deriv
                    # print(last_deriv.shape)
                    layer.backpropagate(last_deriv, learning_rate)
                    prev_layer = layer
                else:
                    print("yeah")
                                
    
    
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

    def save_model(self, filename, indent):
        ind = ""
        for i in range(indent):
            ind+= " "
        # print(ind)
        # print(self.input_shape.__class__)
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