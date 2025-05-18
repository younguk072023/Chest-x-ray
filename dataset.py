from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_loaders(data_dir,batch_size):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    
    #학습
    train_data = datasets.ImageFolder(f"{data_dir}/train",transform=transform)
    #검증
    val_data=datasets.ImageFolder(f"{data_dir}/val", transform=transform)
    #테스트
    test_data = datasets.ImageFolder(f"{data_dir}/test",transform=transform)

    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    print(train_loader.dataset.class_to_idx)



    return train_loader, val_loader, test_loader