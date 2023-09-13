import cv2
from neuralnet import ConvLayer
from neuralnet import Pooling
from neuralnet import Flatten
from neuralnet import Dense
import numpy as np

image = cv2.imread("251.jpeg")


conv = ConvLayer(
    input_shape=image.shape,
    padding=0,
    num_filters=1,
    filter_size=(3, 3),
    stride=2,
    detector_function="relu",
)
pool = Pooling(mode="avg", pool_size=(2, 2), stride=2)

conved = conv.forward_propagate(image)

flatten = Flatten()
flat = flatten.forward_propagate(conved)

dense = Dense(10, flat.size,'relu')
densed = dense.forward_propagate(flat)

# conved = pool.forward_propagate(conved)
print(image.shape)
print(conved.shape)
print(flat.shape)
print(densed.shape)

cv2.imshow("image", conved)

# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()
