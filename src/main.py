import cv2
from neuralnet import NN, ConvLayer
from neuralnet import Pooling
from neuralnet import Flatten
from neuralnet import Dense
import numpy as np

image = cv2.imread("251.jpeg")


#Build the model
model = NN(image.shape)
# print(image.shape)
model.add(ConvLayer(
    input_shape=image.shape,
    padding=0,
    num_filters=1,
    filter_size=(3, 3),
    stride=2,
    detector_function="relu",
))
model.add(Pooling(
    mode = "max", pool_size = (2,2), stride = 2
))
model.add(Flatten())

flat_shape = model.layers[2].feature_map_shape
model.add(Dense(1024, flat_shape,'relu'))

model.add(Dense(64, 1024,'relu'))

flat_shape = model.layers[4].feature_map_shape
model.add(Dense(1, 64,'sigmoid'))

result = model.forward_propagate(image)

if(result <= 0.5):
    category = "Bear"
else:
    category = "Panda"

print(f"Hasil prediksi: {category}")

# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()
