from torchvision import models  #transfer learning 모델 (ResNet, VGG, EfficientNet)
import torch.nn as nn

# 클래스 2로 설정하여 정상/비정상을 분류 모델
def get_resnet18(num_classes=2):    
#pretrained는 기본적인 패턴은 이미 학습돼 있어서 적은 데이터로도 좋은 성능을 끌어올리기 위해 True로 설정
    model = models.resnet18(pretrained=True)
#마지막 fc를 보고 즉, 512개의 숫자로 된 요약된 특징 벡터를 보고 정상인지 비정상인 분류를 진행하는 것.
    model.fc = nn.Linear(512,num_classes)
    return model

