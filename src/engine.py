'''
epoch당 돌아가는 로직
train, val
'''
import torch
from tqdm import tqdm
from .utils import calculate_all_metrics

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_acc =0, 0
    tbar = tqdm(loader)

    for inputs, labels in tbar:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics = calculate_all_metrics(outputs, labels)

        current_acc = metrics['accuracy']
        total_loss += loss.item()
        total_acc += current_acc
        tbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_acc:.4f}")

    return total_loss / len(loader), total_acc/len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    val_loss, val_acc = 0,0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            metrics = calculate_all_metrics(outputs, labels)
            current_acc = metrics['accuracy']

            val_loss += loss.item()
            val_acc += current_acc
    
    return val_loss / len(loader), val_acc / len(loader)
