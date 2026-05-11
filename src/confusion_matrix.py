import torch
import matplotlib.pyplot as plt
import os

from src.dataset import get_loaders
from models.SimpleNet import SimpleNet
from models.VGGNet import VGGNet
from models.ResNet import ResNet
from models.MobileNet import MobileNet
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#혼동행렬 
def evaluate_on_test(data_dir, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_loaders(data_dir, batch_size=32)

    model = MobileNet(in_channels=3, num_classes=2).to(device)
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth"), map_location=device, weights_only=True)) #학습할 때 저장해 두었던 가장 best 불러옴
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NORMAL', 'PNEUMONIA'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix on Test Set")

    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f" Confusion matrix saved to: {save_path}")

    plt.show()

    print("\n Test Set Evaluation Metrics:")
    print(f" - Accuracy : {accuracy_score(all_labels, all_preds):.4f}")
    print(f" - Precision: {precision_score(all_labels, all_preds):.4f}")
    print(f" - Recall   : {recall_score(all_labels, all_preds):.4f}")
    print(f" - F1 Score : {f1_score(all_labels, all_preds):.4f}")