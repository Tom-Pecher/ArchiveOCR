
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import os
from src.utils import config, ProfessionalDigitClassifier, set_random_seeds

# ================================================================
# DATA PREPROCESSING AND AUGMENTATION PIPELINE
# ================================================================
def get_data_transforms():
    """
    Create data transformation pipelines for training and testing.
    Includes normalization based on MNIST dataset statistics.
    """
    # Training transforms with light augmentation for better generalization
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=5),  # Slight rotation for robustness
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Small translations
        transforms.ToTensor(),
        transforms.Normalize((config.NORMALIZE_MEAN,), (config.NORMALIZE_STD,))
    ])

    # Test transforms - only normalization, no augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((config.NORMALIZE_MEAN,), (config.NORMALIZE_STD,))
    ])

    return train_transform, test_transform

def load_datasets():
    """Load and prepare MNIST datasets with appropriate transforms."""
    train_transform, test_transform = get_data_transforms()

    # Create data directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)

    # Load datasets
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)

    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")

    return train_loader, test_loader

# ================================================================
# PROFESSIONAL TRAINING INFRASTRUCTURE
# ================================================================
class TrainingMetrics:
    """Track and manage training metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.losses = []
        self.accuracies = []
        self.learning_rates = []

    def update(self, loss, accuracy, lr):
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        self.learning_rates.append(lr)

def train_epoch(model, train_loader, optimizer, criterion, scheduler, device, epoch):
    """
    Train the model for one epoch with detailed logging.

    Returns:
        avg_loss: Average training loss for the epoch
        accuracy: Training accuracy for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, target)

        # Backward pass and optimization
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Update metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        # Print progress
        if batch_idx % config.PRINT_FREQ == 0:
            current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            print(f'Epoch: {epoch} [{batch_idx * len(data):5d}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):5.1f}%)] '
                  f'Loss: {loss.item():.6f} LR: {current_lr:.6f}')

    # Update learning rate scheduler
    if scheduler:
        scheduler.step()

    epoch_time = time.time() - start_time
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total

    print(f'Epoch {epoch} Training - Loss: {avg_loss:.4f}, '
          f'Accuracy: {accuracy:.2f}%, Time: {epoch_time:.1f}s')

    return avg_loss, accuracy
