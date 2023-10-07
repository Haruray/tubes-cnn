from neuralnet import NN


class Trainer:
    def __init__(self, model:NN, epoch: int, learning_rate: float, input, label):
        self.model = model
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.input = input
        self.label = label
        if(len(input) != len(label)):
            raise Exception("the input length does not compatible with the label")
    
    def fit(self):
        for epoch in range(self.epoch):
            for i in range(len(self.input)):
                self.model.backpropagate(self.input[i], self.label[i], self.learning_rate)