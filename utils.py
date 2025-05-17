import torch

def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs,1)
    return (preds == labels).float().mean().item()
