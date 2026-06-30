import torch
import torchvision
import torchvision.transforms as transforms
from models.resnet import resnet18

def test_transaxx_cifar10():
    """
    test resnet from adapt in transaxx
    """
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # set the transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # load the dataset
    print("load CIFAR-10 dataset...")
    test_set = torchvision.datasets.CIFAR10(root='./datasets/cifar10_data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

    # set the model
    print("Initialization approximate ResNet18...")
    model = resnet18(num_classes=10).to(device)
    model.eval()

    print("compile and load CUDA-core (JIT) for TransAxx-Layer...")
    for module in model.modules():
        if type(module).__name__ == 'AdaptConv2D':
            module.set_axx_kernel()
    print("Cuda core is ready")
    correct = 0
    total = 0
    
    print("start evaluation...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    test_transaxx_cifar10()
