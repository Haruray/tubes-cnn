import numpy as np

class Evaluator():
    def __init__(self,num_classes:int, labels:list, target_labels:list) -> None:

        assert(len(labels) == len(target_labels))
        self.num_classes = num_classes

        self.conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for i in range(len(target_labels)):
            self.conf_matrix[target_labels[i]][labels[i]] += 1

        self.recall_mat = np.zeros(self.num_classes)
        self.precision_mat = np.zeros(num_classes)
        
        print(self.conf_matrix)
    
    def accuracy(self):

        correct_predictions = np.trace(self.conf_matrix)
        total_predictions = np.sum(self.conf_matrix)
        accuracy = correct_predictions / total_predictions
        print(f'Accuracy: {accuracy * 100:.2f}%')

    def recall(self):

        for i in range(self.num_classes):
            true_positives = self.conf_matrix[i, i]
            false_negatives = np.sum(self.conf_matrix[i, :]) - true_positives
            self.recall_mat[i] = true_positives / (true_positives + false_negatives) * 100

            # Calculate the average self.recall
            average_recall = np.mean(self.recall_mat)

            # print("Recall for each class:")
            # for class_idx, class_recall in enumerate(self.recall_mat):
            #     print(f"Class {class_idx}: {class_recall:.2f}")
        
        print(f"Average Recall: {average_recall:.2f}%")
    
    def precision(self):
        
        for i in range(self.num_classes):
            true_positives = self.conf_matrix[i, i]
            false_positives = np.sum(self.conf_matrix[:, i]) - true_positives
            self.precision_mat[i] = true_positives / (true_positives + false_positives) * 100

        average_precision = np.mean(self.precision_mat)
        print(f"Average Precision: {average_precision:.2f}%")
    
    def f1score(self):
        f1score_mat = np.zeros(self.num_classes)

        for i in range(self.num_classes):
            if self.precision_mat[i] == 0 or self.recall_mat[i] == 0:
                f1score_mat[i] = 0
            else:
                f1score_mat[i] = 2 * (self.precision_mat[i] * self.recall_mat[i]) / (self.precision_mat[i] + self.recall_mat[i])

        average_f1score = np.mean(f1score_mat)
        print(f'Average F1 Score: {average_f1score:.2f}%')

# eval = Evaluator(3, [0,2,1,2], [1,2,0,2])
# eval.accuracy()
# eval.recall()
# eval.precision()
# eval.f1score()