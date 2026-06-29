import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T 
import timm 

import sys 
import os 
from torch.utils.data import DataLoader 

# add main direc to python to find classification and layers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

from classification.utils import evaluate_cifar100, cifar100_data_loader, replace_linear_layers, collect_stats, compute_amax
from layers.adapt_linear_layer import AdaPT_Linear

def retrain_model(model, data_path, device, epochs=20):
    """
    retraining loop to recover the accuracy.
    """
    print("\nstart approximation-aware retraining")

    # setup the train data for retraining
    transform_train = T.Compose([
        T.Resize((224, 224)),
        T.RandomCrop(224, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2675, 0.2761)),
    ])

    train_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
    train_dl = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=6)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    # set model to training mode
    model.train()
    for epoch in range(epochs):
        print(f"epoch {epoch+1}/{epochs}")
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
                print(f"batch{i+1}/{len(train_dl)} - loss: {running_loss / 100:.4f}")
                running_loss = 0.0

    print("retraining finished.")

def main():
    """
    evaluate the ViT model on CIFAR100 with TransAxx approximation, calibration and retraining
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # load test and calibration data
    print("load CIFAR100 detaset")
    data_path = "/workspace/datasets/cifar100_data"
    val_data, calib_data = cifar100_data_loader(data_path=data_path, batch_size=128)

    # load model
    print("load pretrained ViT_Tiny model")
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=100)
    model = model.to(device)

    # evaluation with exat math.
    print("evaluation with exat math")
    top1_exat = evaluate_cifar100(model, val_data, device=device)
    print(f"exact top1 accuracy: {top1_exat:.2f}%.\n")

    # TransAxx approximation
    print("TransAxx approximation")
    print("replace the nn.Linear layers with Adapt_Linear")

    total_macs = [0]
    total_params = [0]
    layer_count = [0]
    returned_power = [0]

    num_linear_layers = 100
    axx_list = [{'axx_mult': 'mul8s_1L2H', 'quant_bits': 8, 'fake_quant': True, 'axx_power': 0.5} for _ in range(num_linear_layers)]
    
    replace_linear_layers(model, AdaPT_Linear, axx_list, total_macs, total_params, layer_count, returned_power=returned_power, initialize=True)
    print(f"scuccfully replaced{layer_count[0]} linear layers")

    # calibration
    print("Calibration")
    print("collecting stats for 8_bit quantization")
    collect_stats(model, calib_data, num_batches=2, device=device)
    compute_amax(model, device=device, method="entropy")

    # Evaluation befor retraining
    print("\n evaluation with approximate hardware befor retraining")
    top1_approx_befor = evaluate_cifar100(model, val_data, device=device)

    # retraing the model for 20 rounds
    retrain_model(model, data_path,device, epochs=20)

    # evaluation after retraining
    print("\nevaluation after retraining")
    top1_approx_after = evaluate_cifar100(model, val_data, device=device)

    print("########## summary ######")
    print(f"accuracy (exat):    {top1_exat:.2f}%")
    print(f"accuracy (befor retarainin):    {top1_approx_befor:.2f}%")
    print(f"accuracy (after retraining):    {top1_approx_after:.2f}%")
    print(f"recovered accuracy:     {top1_approx_after - top1_approx_befor:.2f}%")

if __name__ == '__main__':
    main()
