import torch
import os
import matplotlib.pyplot as plt
import glob     #하위 폴더까지 자동 탐색하는 모듈
import random   #랜덤으로 몇개만 추출해서 결과 확인
import seaborn as sns
import matplotlib.font_manager as fm
import platform
from torchvision import transforms
from PIL import Image
from model import get_resnet18


# 폰트 설정 : 한국어 깨짐 방지
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False 


class_names = ['NORMAL', 'PNEUMONIA']

predict_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict_images(predict_dir, model_path, device='cuda' if torch. cuda.is_available() else 'cpu' ):
    model = get_resnet18(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    image_files = glob.glob(os.path.join(predict_dir, '**', '*.jpeg'), recursive=True)
    if not image_files:
        print("이미지가 없습니다!")
        return 
    
    # 전체 결과 저장용 리스트
    y_true=[]
    y_pred=[]
    results = []

    for img_path in image_files:
        # 폴더명 재정의 
        label= 1 if 'PNEUMONIA' in img_path.upper() else 0
        image = Image.open(img_path).convert('RGB')
        input_tensor=predict_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred = torch.max(outputs, 1)
            pred = pred.item()

        y_true.append(label)
        y_pred.append(pred)
        results.append((img_path, pred))

    # 랜덤 10장 시각화
    sample_images = random.sample(results, k=min(10, len(results)))
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Random 10 Predictions", fontsize=16)

    for ax, (img_path, pred) in zip(axes.flatten(), sample_images):
        image = Image.open(img_path).convert('RGB')
        ax.imshow(image)
        ax.set_title(f"예측: {class_names[pred]}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    predict_dir = r"C:\Users\AMI-DEEP3\Desktop\chest\chest_xray\test"
    model_path = r"C:\Users\AMI-DEEP3\Desktop\chest\weights\best_model.pth"  # 저장된 모델 경로

    predict_images(predict_dir, model_path)


