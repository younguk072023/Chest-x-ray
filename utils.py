import torch
import torch.nn.functional as F

#분류 정확도, precision, recall, f1-score

def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs,1)
    return (preds == labels).float().mean().item()

def calculate_precision(outputs, labels, threshold=0.5):
    probs = F.softmax(outputs, dim=1)[:, 1]  # pneumonia 확률만 추출
    preds = (probs >= threshold).int()
    labels = labels.int()

    TP = ((preds == 1) & (labels == 1)).sum().item()
    FP = ((preds == 1) & (labels == 0)).sum().item()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    return precision

def calculate_recall(outputs, labels, threshold=0.5):
    probs = F.softmax(outputs, dim=1)[:, 1]
    preds = (probs >= threshold).int()
    labels = labels.int()

    TP = ((preds == 1) & (labels == 1)).sum().item()
    FN = ((preds == 0) & (labels == 1)).sum().item()

    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    return recall

def calculate_f1_score(outputs, labels, threshold=0.5):
    precision = calculate_precision(outputs, labels, threshold)
    recall = calculate_recall(outputs, labels, threshold)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1
