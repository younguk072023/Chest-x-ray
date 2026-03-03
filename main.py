import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from configs.config import Config
from src.dataset import get_loaders
from src.model import get_model
from src.engine import train_one_epoch, validate
from src.visualizer import plot_results


def main():

    model_save_dir = os.path.join(Config.save_dir, Config.MODEL_NAME)
    os.makedirs(model_save_dir, exist_ok=True)

    train_loader, val_loader, _ = get_loaders(Config.data_dir, Config.batch_size)

    print(f"Model: {Config.MODEL_NAME}, Batch_size: {Config.batch_size}, LR: {Config.lr}, Epochs: {Config.epoch}")
    model = get_model(num_classes=2).to(Config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)

    best_val_loss = float("inf")
    early_stop_counter = 0
    history = {'train_loss':[], 'val_loss':[], 'train_acc':[],'val_acc':[]}

    for epoch in range(Config.epoch):
        print(f"\nEpoch {epoch+1}/{Config.epoch}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, Config.device)
        val_loss, val_acc = validate(model, val_loader, criterion, Config.device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_loss < best_val_loss:
            
            save_path = os.path.join(model_save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)

            best_val_loss = val_loss
            early_stop_counter = 0
    
            print("Best Model Saved.")
            print(f"train loss: {train_loss:.4f} val Loss: {val_loss:.4f}")
        else:
            early_stop_counter += 1
            print(f"train loss: {train_loss:.4f} val Loss: {val_loss:.4f}")
            if early_stop_counter >= Config.patience:
                print("Early Stopping Triggered.")
                break
                
    plot_results(history, model_save_dir)

if __name__ == "__main__":
    main()



