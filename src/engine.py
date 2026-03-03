'''
epoch당 돌아가는 로직
train, val
'''

import torch
from tqdm import tqdm
from .utils import calculate_accuracy

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

        acc = calculate_accuracy(outputs, labels)
        total_loss += loss.item()
        total_acc += acc
        tbar.set_postfix(loss=loss.item(), acc=acc)

    return total_loss / len(loader), total_acc/len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    val_loss, val_acc = 0,0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)

            val_loss += loss.item()
            val_acc += acc
    
    return val_loss / len(loader), val_acc / len(loader)
