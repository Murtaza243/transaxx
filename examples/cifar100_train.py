import torch
import torch.nn as nn
import torch.optim as optim
import timm 

import sys
import os

# add main direct to find datasets
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.cifar100_loader import get_cifar100_dataloader

def main():
    """
    exact baseline training for ViT on CIFAR100 (to get the exact weights)
    """
    # setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # load data
    print("load CIFAR100 dataloader")
    train_dl, test_dl = get_cifar100_dataloader(batch_size=128, img_size=224, num_workers=10)

    # load model
    print("load pretrained ViT_Tiny_model")
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=100)
    model = model.to(device)

    # setup training config
    epochs = 50
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_accuracy = 0.0
    # path for the weights
    save_path = "/workspace/examples/vit_tiny_cifar100_weights.pt"
    
    print("\n##### start the baseline training ######")
    for epoch in range(epochs):
        print(f"\nepoch {epoch+1}/{epochs}")
        # train
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_dl):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f" batch {i+1}/{len(train_dl)} - loss: {running_loss / 100:.4f}")
                running_loss = 0.0

        # step the learning rate scheduler after each epoch
        scheduler.step()

        # Evaluate
        model.eval()
        correct = 0
        total = 0 
        with torch.no_grad():
            for images, labels in test_dl:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct /total
        print(f"validation accuracy: {accuracy:.2f}#")

        #save the model if it is the best one we have seen so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"new best model. saving to {save_path}")
            torch.save(model.state_dict(), save_path)

    print("\nBaseline training finished.")
    print(f"absolute best accuracy: {best_accuracy:.2f}%.")
    print(f"model saved at: {save_path}")

if __name__ == '__main__':
    main()
