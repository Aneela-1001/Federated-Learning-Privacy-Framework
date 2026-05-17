"""
Shadow Model Training for Black-Box Membership Inference Attack (MIA).

In black-box MIA, the attacker doesn't have access to the target model's training data.
Instead, the attacker trains "shadow models" on auxiliary data that mimics the target model's
training distribution. These shadow models help the attacker understand:
1. How the target model behaves on training data (members) vs. non-training data (non-members)
2. What features distinguish member predictions from non-member predictions

This module implements shadow model training using auxiliary data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Tuple, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ShadowModel(nn.Module):
    """
    Shadow model architecture - should match the target model architecture.
    For MNIST-like classification, uses a simple feedforward network.
    """
    
    def __init__(self, num_features: int = 784, num_classes: int = 10):
        """
        Initialize shadow model.
        
        Args:
            num_features: Number of input features (784 for 28x28 MNIST images)
            num_classes: Number of output classes (10 for MNIST digits)
        """
        super(ShadowModel, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the shadow model.
        
        Args:
            x: Input tensor (batch_size, num_features)
            
        Returns:
            Output logits (batch_size, num_classes)
        """
        # Flatten input if needed
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def train_shadow_model(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 0.01,
    device: torch.device = None
) -> nn.Module:
    """
    Train a shadow model on auxiliary data.
    
    This simulates training a model similar to the target model, but on different data.
    The shadow model helps the attacker understand model behavior patterns.
    
    Args:
        model: Shadow model to train
        train_loader: DataLoader with training data (auxiliary data)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on (CPU/GPU)
        
    Returns:
        Trained shadow model
    """
    if device is None:
        device = torch.device("cpu")
    
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    logger.info(f"Training shadow model for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    model.eval()
    logger.info("Shadow model training completed")
    
    return model


def create_shadow_datasets(
    member_data: torch.Tensor,
    member_labels: torch.Tensor,
    non_member_data: torch.Tensor,
    non_member_labels: torch.Tensor,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    Create shadow model training datasets from member and non-member data.
    
    In black-box MIA:
    - Member data: Data that was used to train the shadow model (simulates target model's training data)
    - Non-member data: Data NOT used to train the shadow model (simulates target model's test/holdout data)
    
    The shadow model is trained only on member data, then used to generate predictions
    on both member and non-member data. This helps the attacker learn patterns.
    
    Args:
        member_data: Training data for shadow model (tensor)
        member_labels: Training labels for shadow model (tensor)
        non_member_data: Holdout data (not used for training) (tensor)
        non_member_labels: Holdout labels (tensor)
        batch_size: Batch size for DataLoaders
        
    Returns:
        Tuple of (member_loader, non_member_loader)
    """
    # Create datasets
    member_dataset = TensorDataset(member_data, member_labels)
    non_member_dataset = TensorDataset(non_member_data, non_member_labels)
    
    # Create data loaders
    member_loader = DataLoader(
        member_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    non_member_loader = DataLoader(
        non_member_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    logger.info(f"Created shadow datasets: {len(member_data)} members, {len(non_member_data)} non-members")
    
    return member_loader, non_member_loader


def generate_shadow_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate predictions from shadow model for attack dataset construction.
    
    This function queries the shadow model (black-box access) and extracts:
    - Prediction probabilities (softmax outputs)
    - Predicted classes
    - True labels
    
    These will be used to construct features for the attack classifier.
    
    Args:
        model: Trained shadow model
        data_loader: DataLoader with data to predict on
        device: Device to run inference on
        
    Returns:
        Tuple of (probabilities, predicted_classes, true_labels)
        - probabilities: (n_samples, n_classes) array of softmax probabilities
        - predicted_classes: (n_samples,) array of predicted class indices
        - true_labels: (n_samples,) array of true class labels
    """
    if device is None:
        device = torch.device("cpu")
    
    model.to(device)
    model.eval()
    
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            
            # Get model predictions (black-box access - only outputs, no internals)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())  # Labels are already on CPU from DataLoader
    
    # Concatenate all batches
    probabilities = np.vstack(all_probs)
    predicted_classes = np.concatenate(all_preds)
    true_labels = np.concatenate(all_labels)
    
    logger.info(f"Generated predictions for {len(probabilities)} samples")
    
    return probabilities, predicted_classes, true_labels


def train_multiple_shadow_models(
    num_shadow_models: int,
    member_data: torch.Tensor,
    member_labels: torch.Tensor,
    non_member_data: torch.Tensor,
    non_member_labels: torch.Tensor,
    num_features: int = 784,
    num_classes: int = 10,
    num_epochs: int = 10,
    batch_size: int = 32,
    device: torch.device = None
) -> List[Tuple[nn.Module, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Train multiple shadow models to create a diverse attack dataset.
    
    Training multiple shadow models helps the attacker:
    1. Learn more robust patterns (ensemble effect)
    2. Handle model variability
    3. Create a larger attack training dataset
    
    Each shadow model is trained on a different subset of auxiliary data,
    then used to generate predictions on both member and non-member data.
    
    Args:
        num_shadow_models: Number of shadow models to train
        member_data: Training data for shadow models
        member_labels: Training labels
        non_member_data: Holdout data (not used for training)
        non_member_labels: Holdout labels
        num_features: Number of input features
        num_classes: Number of output classes
        num_epochs: Number of training epochs per shadow model
        batch_size: Batch size for training
        device: Device to train on
        
    Returns:
        List of tuples, each containing:
        - shadow_model: Trained shadow model
        - member_probs: Predictions on member data
        - member_preds: Predicted classes on member data
        - non_member_probs: Predictions on non-member data
        - non_member_preds: Predicted classes on non-member data
    """
    if device is None:
        device = torch.device("cpu")
    
    results = []
    
    # Split member data into subsets for each shadow model
    total_member_samples = len(member_data)
    samples_per_model = total_member_samples // num_shadow_models
    
    logger.info(f"Training {num_shadow_models} shadow models...")
    
    for i in range(num_shadow_models):
        logger.info(f"Training shadow model {i+1}/{num_shadow_models}...")
        
        # Get subset of member data for this shadow model
        start_idx = i * samples_per_model
        end_idx = start_idx + samples_per_model if i < num_shadow_models - 1 else total_member_samples
        
        shadow_member_data = member_data[start_idx:end_idx]
        shadow_member_labels = member_labels[start_idx:end_idx]
        
        # Create shadow model
        shadow_model = ShadowModel(num_features=num_features, num_classes=num_classes)
        
        # Create data loaders
        shadow_member_loader, shadow_non_member_loader = create_shadow_datasets(
            shadow_member_data,
            shadow_member_labels,
            non_member_data,
            non_member_labels,
            batch_size=batch_size
        )
        
        # Train shadow model
        shadow_model = train_shadow_model(
            shadow_model,
            shadow_member_loader,
            num_epochs=num_epochs,
            device=device
        )
        
        # Generate predictions on both member and non-member data
        member_probs, member_preds, member_true_labels = generate_shadow_predictions(
            shadow_model,
            shadow_member_loader,
            device=device
        )
        
        non_member_probs, non_member_preds, non_member_true_labels = generate_shadow_predictions(
            shadow_model,
            shadow_non_member_loader,
            device=device
        )
        
        results.append((
            shadow_model,
            member_probs,
            member_preds,
            member_true_labels,
            non_member_probs,
            non_member_preds,
            non_member_true_labels
        ))
    
    logger.info(f"All {num_shadow_models} shadow models trained successfully")
    
    return results
