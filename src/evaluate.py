import evaluate
import torch


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = torch.argmax(predictions, dim=1)
    return accuracy.compute(predictions=predictions, references=labels)
