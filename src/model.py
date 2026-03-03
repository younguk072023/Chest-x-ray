from torchvision import models
import torch.nn as nn
from configs.config import Config

def get_model(model_name = Config.MODEL_NAME, num_classes=2):
    """
    4가지 주요 모델 비교를 위한 통합 함수
    """
    # ResNet 계열
    if model_name == "resnet18":    #18,50 ...
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # DenseNet 계열 
    elif model_name == "densenet121":   #121, 161, 169 ...
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    # EfficientNet 계열 
    elif model_name == "efficientnet_b0":   #0, 1, 2, 3 ...
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # MobileNet 계열 
    elif model_name == "mobilenet_v2":    #mobilenet_v2, v3
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
        
    else:
        raise ValueError(f"모델명 확인: {model_name}")
        
    return model