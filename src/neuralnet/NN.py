from neuralnet import Layer
import numpy as np
import json


class NN:
    def __init__(self, input_shape: tuple, layers: Layer = []):
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
        prev_layer = None
        for layer in self.layers:
            image = layer.forward_propagate(image)
            layer.last_layer = prev_layer
            prev_layer = layer
        return image

    def predict(self, images: np.ndarray):
        if len(images.shape) == 3:
            return self.forward_propagate(images)
        elif len(images.shape) == 4:
            result = []
            for image in images:
                result.append(self.forward_propagate(image))
            result = list(np.array(result).flatten())
            result = map(self.output_to_label, result)
            return list(result)
        else:
            raise Exception("The image shape is not valid")

    def output_to_label(self, output):
        if output > 0.5:
            return 1
        else:
            return 0

    def calculate_derr_error(self, layer: Layer, preds: np.ndarray, labels: np.ndarray):
        # source : https://www.pinecone.io/learn/cross-entropy-loss/
        if len(preds) != len(labels):
            raise Exception(
                f"The label: {len(labels)} and prediction: {len(preds)} does not have same length"
            )
        return preds - labels

    def backpropagate(self, input, label, learning_rate):

        layers_count = len(self.layers)
        if layers_count <= 0:
            raise Exception("There is no layers to backpropagate")
        result = self.forward_propagate(input)
        prev_layer = self.layers[layers_count - 1]  # final layer
        last_deriv = None
        for i in range(layers_count - 1, -1, -1):
            layer = self.layers[i]
            # CASE OUTPUT LAYER (OBVIOUSLY DENSE LAYER)
            if i == layers_count - 1:
                layer.last_input = np.expand_dims(layer.last_input, axis=1).T
                last_deriv = self.calculate_derr_error(
                    prev_layer, result, label
                )  # dE/dNet
                last_deriv = np.expand_dims(last_deriv, axis=1)
                layer.backpropagate(last_deriv, learning_rate)  # dE/dNet * dNet/dW

            # CASE HIDDEN LAYER
            else:
                # CASE TYPE DENSE
                if layer.type == "dense":
                    """
                    Rumus general
                    misal current layer adalah i, dan last layer adalah j dan k adalah layer setelah i
                    maka backprop layer i adalah :
                    dE/dWi = dE/dNetj * dNetj/dActivation * dActivation/dNeti * dNeti/dWi

                    Maka bisa disimpulkan bahwa rumus sederhananya :
                    dE/dWi = dE/dNetj * Wj * derivActivationFunc(Netj) * Netk

                    dE/dNetj = last_deriv --> sudah diketahui
                    Wj = prev_layer.weights --> sudah diketahui
                    Netk = layer.last_input --> sudah diketahui
                    derivActivationFunc(Netj) = layer.detector_function.deriv(prev_layer.last_input) --> sudah diketahui

                    """
                    layer.last_input = np.expand_dims(
                        layer.last_input, axis=1
                    ).T  # biar bisa dikalikan dengan last_deriv
                    last_deriv = np.dot(last_deriv, prev_layer.weights)
                    last_deriv = last_deriv * layer.detector_function.deriv(
                        prev_layer.last_input
                    )
                    layer.backpropagate(last_deriv, learning_rate)
                elif layer.type == "flatten":
                    # mengubah bentuk flatten ke bentuk sebelumnya, yaitu last_input
                    last_deriv = np.dot(last_deriv, prev_layer.weights)
                    last_deriv = layer.backpropagate(last_deriv, learning_rate)
                else:
                    last_deriv = layer.backpropagate(last_deriv, learning_rate)
            prev_layer = layer

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def save_model(self, filename, indent):
        ind = ""
        for i in range(indent):
            ind += " "
        # print(ind)
        # print(self.input_shape.__class__)
        with open(filename, "w") as f:
            f.write("{\n")
            f.write(f"""{ind}"input_shape": {list(self.input_shape)},\n""")
            f.write(f"""{ind}"layers": [\n""")
            for i in range(len(self.layers)):
                f.write(f"""{ind}{ind}{self.layers[i]}""")
                if i == len(self.layers) - 1:
                    f.write("\n")
                else:
                    f.write(",\n")
            f.write(f"""{ind}]\n""")
            f.write("}\n")
