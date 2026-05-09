'''
그래프 시각화
loss, acc 그래프

+ 혼동행렬 figure
'''

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_results(history, save_dir):

    # Loss Curve
    plt.figure(figsize=(8, 6))
    plt.plot(history['train_loss'], label="Train Loss", color='blue')
    plt.plot(history['val_loss'], label="Val Loss", color='orange')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=300)
    plt.close()

    # Accuracy Curve
    plt.figure(figsize=(8, 6))
    plt.plot(history['train_acc'], label='Train Acc', color='green')
    plt.plot(history['val_acc'], label='Val Acc', color='red')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=300)
    plt.close()

def save_confusion_matrix(y_true, y_pred, class_names, save_dir):
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Final Confusion Matrix')
    plt.ylabel('Actual Category')
    plt.xlabel('Predicted Category')
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300)
    plt.close()
    print(f"모든 시각화 결과가 {save_dir}에 저장되었습니다.")