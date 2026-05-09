import os 
import torch
import numpy as np
import cv2
from configs.config import Config
from models.SimpleNet import SimpleNet
from src.dataset import get_loaders
from src.utils import calculate_all_metrics
from src.visualizer import save_confusion_matrix
from src.gradcam import GradCam # 💡 GradCam 클래스 임포트

def predict():
    device = Config.device
    model_path = os.path.join(Config.save_dir, Config.MODEL_NAME, "best_model.pth")
    save_dir = os.path.join(Config.save_dir, Config.MODEL_NAME)
    gradcam_save_dir = os.path.join(save_dir, "gradcam_results")
    os.makedirs(gradcam_save_dir, exist_ok=True)

    _, _, test_loader = get_loaders(Config.data_dir, Config.batch_size)

    model = SimpleNet(in_channels=3, num_classes=2).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"📦 Model loaded: {model_path}")
    else:
        return
    
    model.eval()

    # 💡 Grad-CAM 설정 (SimpleNet의 마지막 Conv 블록인 block3 지정)
    # 모델 구조에 따라 target_layer를 적절히 조절하세요.
    gcam = GradCam(model=model, target_layer=model.block3)

    all_outputs = []
    all_labels = []

    print("🚀 Testing & Generating Grad-CAM...")
    
    # Grad-CAM은 역전파(gradient)를 이용하므로 torch.no_grad()를 쓰면 안 됩니다.
    # 대신 모델의 가중치는 업데이트되지 않도록 주의해야 합니다.
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        
        # 1. Grad-CAM 마스크 생성 (배치의 첫 번째 이미지에 대해 예시로 수행)
        # 모든 이미지에 대해 다 하려면 반복문을 하나 더 돌려야 합니다.
        for j in range(images.size(0)):
            img_tensor = images[j].unsqueeze(0) # [1, 3, H, W]
            mask = gcam(img_tensor, target_index=None) # 히트맵 마스크 생성
            
            # 시각화를 위해 원본 이미지 복구 (Normalize 역산 필요할 수 있음)
            orig_img = images[j].cpu().permute(1, 2, 0).numpy()
            orig_img = (orig_img * 255).astype(np.uint8)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
            
            # 히트맵 합성
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            cam_result = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)
            
            # 파일 저장 (예: result_0_normal.png)
            label_name = 'PNEUMONIA' if labels[j].item() == 1 else 'NORMAL'
            cv2.imwrite(os.path.join(gradcam_save_dir, f"cam_{i}_{j}_{label_name}.png"), cam_result)

        # 평가를 위한 결과 수집
        with torch.no_grad():
            outputs = model(images)
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())
        
        if i == 5: break # 💡 너무 많이 저장되지 않도록 5배치까지만 수행 (조절 가능)

    # ... (이하 동일: metrics 계산 및 Confusion Matrix 저장 로직) ...
    final_outputs = torch.cat(all_outputs, dim=0)
    final_labels = torch.cat(all_labels, dim=0)
    _, final_preds = torch.max(final_outputs, 1)

    metrics = calculate_all_metrics(final_outputs, final_labels)
    print(f"\n✅ Grad-CAM 결과가 {gradcam_save_dir}에 저장되었습니다.")
    
    save_confusion_matrix(final_labels.numpy(), final_preds.numpy(), 
                          ['NORMAL', 'PNEUMONIA'], save_dir)

if __name__ == '__main__':
    predict()