import cv2
from neuralnet import NN, ConvLayer
from neuralnet import Pooling
from neuralnet import Flatten
from neuralnet import Dense
from neuralnet import load_model
import numpy as np
from neuralnet import Trainer
from neuralnet import Preprocess

data_preprocess = Preprocess("../data", False)
train, test = data_preprocess.get_data(shuffle=True)

# Build the model
newModel = load_model("yeah.json")

trainer = Trainer(
    newModel,
    1,
    0.1,
    train.get_images(),
    train.get_labels(),
    test_input=test.get_images(),
    test_label=test.get_labels(),
)
# trainer.fit()
# print(newModel.layers[4].weights)

# image = cv2.imread("251.jpeg")
# model = NN(image.shape)
# model.add(
#     ConvLayer(
#         input_shape=image.shape,
#         padding=0,
#         num_filters=1,
#         filter_size=(3, 3),
#         stride=2,
#         detector_function="relu",
#     )
# )
# model.add(Pooling(mode="max", pool_size=(2, 2), stride=2))
# model.add(Flatten())

# flat_shape = model.layers[2].feature_map_shape
# model.add(Dense(64, flat_shape, "relu"))

# model.add(Dense(1, 64, "sigmoid"))
# model.save_model("small.json", 4)
