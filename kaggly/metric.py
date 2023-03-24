import evaluate
import numpy as np

class KagglyEvaluate():
    accuracy = evaluate.load("accuracy")
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.accuracy.compute(predictions=predictions, references=labels)