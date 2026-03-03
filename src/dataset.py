from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_loaders(data_dir,batch_size):
    #transform은 이미지에 적용하는 전처리 또는 변형 작업업
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  #이미지를 좌우로 무작위로 뒤집음
        transforms.RandomRotation(10),      #이미지를 10도 이내로  무작위 회전
        transforms.RandomResizedCrop(224, scale=(0.8,1.0)), #resize
        transforms.ColorJitter(brightness = 0.1, contrast=0.1), #밝기, 대비를 무작위로 약간씩 변환환
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    #학습
    train_data = datasets.ImageFolder(f"{data_dir}/train",transform=train_transform)
    #검증
    val_data=datasets.ImageFolder(f"{data_dir}/val", transform=val_test_transform)
    #테스트
    test_data = datasets.ImageFolder(f"{data_dir}/test",transform=val_test_transform)

    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    print(train_loader.dataset.class_to_idx)


    return train_loader, val_loader, test_loader