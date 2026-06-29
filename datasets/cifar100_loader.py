import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_cifar100_dataloader(batch_size=128, img_size=224, num_workers=4):
    """
    loader the CIFAR100 Dataset and modify it for ViT.
    
    Args:
        batch_size (int): batch_size for training.
        img_size (int): image size.
        num_workers (int): number of threads for loading. 

    returns:
        train_loader, test_loader: pytorch DataLoader objects.
    """

    # Data modification (transformetion)
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2568, 0.2761)),
    ])

    print("load CIFAR100 Train-Set")
    trainset = torchvision.datasets.CIFAR100(
        root='./datasets/cifar100_data', train=True, download=True, transform=transform_train
    )

    print("load CIFAR100 Test-Set")
    testset = torchvision.datasets.CIFAR100(
        root='./datasets/cifar100_data', train=False, download=True, transform=transform_train
    )

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, test_loader


if __name__ == '__main__':
    train_dl, test_dl = get_cifar100_dataloader(batch_size=16)
    images, labels = next(iter(train_dl))
    print(f"image batch shape: {images.shape} ### [16, 3, 224, 224]")
    print(f"label batch shape: {labels.shape} ### [16]")
