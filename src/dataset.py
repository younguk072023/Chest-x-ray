from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


class ImageFolderWithPaths(datasets.ImageFolder):

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path = self.samples[index][0]

        return image, label, path


def get_loaders(data_dir, batch_size, return_test_paths=False):

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_data = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )

    val_data = datasets.ImageFolder(
        root=val_dir,
        transform=val_test_transform
    )

    if return_test_paths:
        test_data = ImageFolderWithPaths(
            root=test_dir,
            transform=val_test_transform
        )
    else:
        test_data = datasets.ImageFolder(
            root=test_dir,
            transform=val_test_transform
        )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader