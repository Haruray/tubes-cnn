import cv2
from neuralnet import ConvLayer

image = cv2.imread("251.jpeg")

conv = ConvLayer(
    input_shape=image.shape, padding=0, num_filters=1, filter_size=(3, 3), stride=1
)

print(conv.forward_propagate(image))