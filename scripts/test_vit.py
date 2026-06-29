import torch 
import timm 

import sys
import os

# find datasets
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.cifar100_loader import get_cifar100_dataloader

def setup_and_test_model():
    """
    Setup a Vision Transformer and test inference with CIFAR100
    """

    # Device Setup
    device = torch.device("cuda")
    print(f"using device: {device}")

    # Get DataLoaders
    train_dl, test_dl = get_cifar100_dataloader(batch_size=128, num_workers=6)
    
    # Load ViT model from timm 
    print("load Vision Transformer: vit_tiny_patch16_224")
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=100)
    
    model = model.to(device)
    model.eval()

    # perform test inference
    print("\nrun test inference.")

    images, labels = next(iter(test_dl))

    images = images.to(device)
    labels = labels.to(device)


    with torch.no_grad():
        outputs = model(images)

    # check the results
    print(f"shape of model outputs: {outputs.shape} ### [128, 100]")

    _, predicted_classes = torch.max(outputs, 1)

    print("\nfirst 5 true labels: ", labels[:5].tolist())
    print("\nfirst 5 predicted labels: ", predicted_classes[:5].tolist())
    print("\nsuccess! model is ready and runnning on GPU.")

if __name__ == '__main__':
    setup_and_test_model()



