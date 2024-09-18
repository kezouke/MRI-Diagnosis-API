import torch
import torch.nn as nn
import torch.optim as optim
from code.utils.get_data_loaders import create_data_loaders, get_project_root
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
from typing import Dict, List
from tqdm import tqdm

# Check for available devices
use_cuda = torch.cuda.is_available()
use_mps = torch.backends.mps.is_available()
device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")


class BrainTumorClassifier(nn.Module):

    def __init__(self, num_classes: int):
        super(BrainTumorClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128 * 18 * 18, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(-1, 128 * 18 * 18)  # Flatten the tensor
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def save_model(model: nn.Module, model_name: str) -> None:
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    model_path = os.path.join(get_project_root(), 'models',
                              f'{model_name}_{timestamp}.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')


def plot_and_save_accuracy(history: Dict[str, List[float]], num_epochs: int,
                           model_name: str) -> None:
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    epochs_range = range(num_epochs)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history['accuracy'], label='Training Accuracy')
    plt.plot(epochs_range,
             history['val_accuracy'],
             label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    plot_path = os.path.join(get_project_root(), 'data',
                             f'{model_name}_accuracy_{timestamp}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f'Accuracy plot saved to {plot_path}')


def plot_and_save_sample_results(test_loader, model, num_samples: int,
                                 model_name: str) -> None:
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    labels = test_loader.dataset.classes
    predicted_labels = [labels[p] for p in all_preds]
    actual_labels = [labels[l] for l in all_labels]

    random_indices = np.random.choice(len(all_preds),
                                      num_samples,
                                      replace=False)

    plt.figure(figsize=(12, 12))
    for i, idx in enumerate(random_indices):
        plt.subplot(5, 5, i + 1)
        img = test_loader.dataset[idx][0].permute(1, 2, 0).numpy()
        # Normalize image data to range [0, 1]
        img = (img - img.min()) / (img.max() - img.min())

        plt.imshow(img)
        plt.title(
            f"Actual: {actual_labels[idx]}\nPredicted: {predicted_labels[idx]}"
        )
        plt.axis('off')

    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    plot_path = os.path.join(get_project_root(), 'data',
                             f'{model_name}_samples_{timestamp}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f'Sample results plot saved to {plot_path}')


def train_model(model: nn.Module,
                train_loader,
                val_loader,
                num_epochs: int = 10) -> Dict[str, List[float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_accuracy = 0.0
    history = {'accuracy': [], 'val_accuracy': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        history['accuracy'].append(epoch_accuracy)
        print(
            f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%'
        )

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation", unit="batch"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        history['val_accuracy'].append(val_accuracy)
        print(
            f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%'
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_model(model, 'model_a')

    return history


def evaluate_test_accuracy(model: nn.Module, test_loader) -> None:
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')


if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_data_loaders()
    original_dataset = train_loader.dataset.dataset
    num_classes = len(original_dataset.classes)

    model = BrainTumorClassifier(num_classes=num_classes).to(device)

    history = train_model(model, train_loader, val_loader, num_epochs=20)
    plot_and_save_accuracy(history, num_epochs=20, model_name='model_a')
    plot_and_save_sample_results(test_loader,
                                 model,
                                 num_samples=10,
                                 model_name='model_a')
    evaluate_test_accuracy(model, test_loader)
