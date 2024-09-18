import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch


def get_project_root():
    """
    Returns the absolute path to the root directory of the project.
    Assumes that the README.md is located in the root.
    
    Returns:
    - root_dir (str): Absolute path to the project root directory.
    """
    # Get the current file path (the path to this file: get_data_loaders.py)
    current_file_path = os.path.dirname(os.path.abspath(__file__))

    # Navigate up to the root directory, which contains 'README.md'
    root_dir = os.path.abspath(os.path.join(current_file_path, '../../'))

    return root_dir


def create_data_loaders(batch_size=32, image_size=150, val_split=0.2):
    """
    Create DataLoaders for training and testing datasets.
    
    Args:
    - train_dir (str): Path to the training data directory.
    - test_dir (str): Path to the testing data directory.
    - batch_size (int): Batch size for DataLoader.
    - image_size (int): Size to which images will be resized.
    
    Returns:
    - train_loader (DataLoader): DataLoader for training data.
    - test_loader (DataLoader): DataLoader for testing data.
    """
    # Get the absolute project root path
    project_root = get_project_root()

    # Define absolute paths for the training and testing datasets
    train_dir = os.path.join(project_root, 'code', 'datasets', 'Training')
    test_dir = os.path.join(project_root, 'code', 'datasets', 'Testing')

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),  # Add vertical flip
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Define transformations for the testing dataset (no augmentations)
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create datasets using ImageFolder
    train_dataset = datasets.ImageFolder(root=train_dir,
                                         transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir,
                                        transform=test_transform)

    # Split the training dataset into training and validation sets
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_split * num_train))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)

    train_loader = DataLoader(dataset=train_subset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2)
    val_loader = DataLoader(dataset=val_subset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=2)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_data_loaders()
    labels = test_loader.dataset.classes
    print(labels)