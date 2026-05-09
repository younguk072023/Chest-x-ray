import torch
import torch.nn.functional as F

#확률 계산 및 판정 내리는 함수
def get_binary_predictions(outputs, threshold=0.5):
    probs = F.softmax(outputs, dim=1)[:, 1]
    preds = (probs >= threshold).long()
    return probs, preds

def calculate_confusion_matrix_values(outputs, labels, threshold=0.5):
    _, preds = get_binary_predictions(outputs, threshold)
    labels = labels.long()

    TP = ((preds == 1) & (labels == 1)).sum().item()
    TN = ((preds == 0) & (labels == 0)).sum().item()
    FP = ((preds == 1) & (labels == 0)).sum().item()
    FN = ((preds == 0) & (labels == 1)).sum().item()

    return TP, TN, FP, FN

def calculate_all_metrics(outputs, labels, threshold=0.5):
    TP, TN, FP, FN = calculate_confusion_matrix_values(outputs, labels, threshold)
    
    total = TP + TN + FP + FN
    
    accuracy = (TP + TN) / total if total > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
    }

    return metrics