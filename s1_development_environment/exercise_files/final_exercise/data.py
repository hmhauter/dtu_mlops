import torch
import os
from torch.utils.data import DataLoader, TensorDataset

def mnist():
    """Return train and test dataloaders for MNIST."""
    local_dir = "/home/hhauter/Documents/W23/MLOps/dtu_mlops/data/corruptmnist"
    train_files = [f for f in os.listdir(local_dir) if f.startswith('train_images')]
    test_files = [f for f in os.listdir(local_dir) if f.startswith('test_images')]

    # train
    train_images = torch.cat([torch.load(f"{local_dir}/train_images_{i}.pt") for i in range(len(train_files))])
    train_targets = torch.cat([torch.load(f"{local_dir}/train_target_{i}.pt") for i in range(len(train_files))])
    train_dataset = TensorDataset(train_images, train_targets)
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # test
    test_images = torch.load(f"{local_dir}/test_images.pt")
    test_targets = torch.load(f"{local_dir}/test_target.pt")
    test_dataset = TensorDataset(test_images, test_targets)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return trainloader, testloader
