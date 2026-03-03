'''
하이퍼파라미터 및 경로 설정
'''

import os
import torch

class Config:

    #data 경로
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_path, 'chest_xray')
    save_dir = os.path.join(base_path, 'weights')

    MODEL_NAME = "mobilenet_v2"     #torchvision model 참고

    epoch=100
    lr=1e-4
    batch_size=32
    patience=10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


 