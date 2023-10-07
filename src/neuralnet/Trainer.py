from neuralnet.NN import NN
from neuralnet.Evaluator import Evaluator


class Trainer:
    def __init__(
        self,
        model: NN,
        epoch: int,
        learning_rate: float,
        input,
        label,
        test_input,
        test_label,
    ):
        self.model = model
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.input = input
        self.label = label
        if len(input) != len(label):
            raise Exception("the input length does not compatible with the label")

    def fit(self):
        print("Training model...")
        for epoch in range(self.epoch):
            print(f"Epoch {epoch+1}/{self.epoch}")
            for i in range(len(self.input)):
                self.model.backpropagate(
                    self.input[i], self.label[i], self.learning_rate
                )
            eval = Evaluator(2, self.model.predict(self.input), self.label.flatten())
            eval.confusion_matrix()
            eval.accuracy()
            eval.precision()
            eval.recall()
            eval.f1score()
        print("Training finished. Saving model...")
        self.model.save_model("trained.json", 4)
