import json

def calculate_metrics(pos_predictions:list, true_labels_path:str):
    """
    Calculate accuracy, precision, recall, and F1-score for the given requirement and the positive predictions from the model.
    Used to compute after all 10K images are passed through a model. Using on a subset will require modifications.

    Args:
    pos_predictions (list): List of image IDs that are predicted as positive by the model.
    """
    with open(true_labels_path, 'r') as f:
        true_labels = json.load(f)
    TP = sum([1 for id in pos_predictions if true_labels[id] == 1])
    FP = len(pos_predictions) - TP
    FN = sum([1 for id in true_labels if true_labels[id] == 1 and id not in pos_predictions])
    TN = len(true_labels) - TP - FN - FP

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    
    return accuracy, precision, recall, f1_score