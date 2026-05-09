import os 
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from configs.config import Config
from src.model import get_model
from src.dataset import get_loaders


def predict():
    device = Config.device
    model_path = os.path.join(Config.save_dir, Config.MODEL_NAME, "best_model.pth")

    _, _, test_loader = get_loaders(Config.data_dir, Config.batch_size)

    model = get_model(num_classes=2).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"load: {model_path}")
    else:
        print("Error: not found model. go to train")
        return
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    print("Testing in progress ...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()