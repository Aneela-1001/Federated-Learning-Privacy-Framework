"""
Attack Model for Black-Box Membership Inference Attack (MIA).

The attack model is a binary classifier that distinguishes between:
- Member samples: Data points that were in the target model's training set
- Non-member samples: Data points that were NOT in the target model's training set

The attack model is trained on features extracted from shadow model predictions.
These features capture patterns that indicate membership (e.g., high confidence on
correct predictions suggests the sample was in training data).

This module implements the attack classifier using logistic regression or MLP.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Tuple, Optional, List
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AttackMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) attack classifier.
    
    This is a small neural network that learns to classify samples as
    members or non-members based on prediction features.
    """
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, num_classes: int = 2):
        """
        Initialize attack MLP.
        
        Args:
            input_dim: Number of input features (from prediction features)
            hidden_dim: Number of hidden units
            num_classes: Number of output classes (2: member/non-member)
        """
        super(AttackMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attack classifier.
        
        Args:
            x: Input features (batch_size, input_dim)
            
        Returns:
            Output logits (batch_size, num_classes)
        """
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class AttackLogisticRegression(nn.Module):
    """
    Logistic Regression attack classifier (simpler alternative to MLP).
    
    This is a linear classifier that can be faster to train and interpret.
    """
    
    def __init__(self, input_dim: int = 10, num_classes: int = 2):
        """
        Initialize logistic regression classifier.
        
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes (2: member/non-member)
        """
        super(AttackLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through logistic regression.
        
        Args:
            x: Input features (batch_size, input_dim)
            
        Returns:
            Output logits (batch_size, num_classes)
        """
        return self.linear(x)


def extract_prediction_features(
    probabilities: np.ndarray,
    predicted_classes: np.ndarray,
    true_labels: np.ndarray,
    loss_values: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Extract features from model predictions for membership inference.
    
    These features capture patterns that distinguish members from non-members:
    - High confidence on correct predictions → likely member
    - Low entropy → model is confident → might be member
    - Large confidence gap → model is certain → might be member
    
    Args:
        probabilities: (n_samples, n_classes) array of prediction probabilities
        predicted_classes: (n_samples,) array of predicted class indices
        true_labels: (n_samples,) array of true class labels
        loss_values: Optional (n_samples,) array of loss values per sample
        
    Returns:
        (n_samples, n_features) array of extracted features
    """
    n_samples = len(probabilities)
    features = []
    
    for i in range(n_samples):
        prob = probabilities[i]
        pred_class = predicted_classes[i]
        true_label = true_labels[i]
        
        # Feature 1: Maximum prediction confidence
        max_confidence = prob.max()
        
        # Feature 2: Prediction entropy (higher entropy = more uncertainty)
        entropy = -np.sum(prob * np.log(prob + 1e-10))
        
        # Feature 3: Correctness (1 if correct, 0 if wrong)
        correctness = 1.0 if pred_class == true_label else 0.0
        
        # Feature 4: Confidence of predicted class
        pred_confidence = prob[pred_class]
        
        # Feature 5: Confidence of true class
        true_class_confidence = prob[true_label]
        
        # Feature 6: Top-2 confidence gap (difference between top-1 and top-2)
        sorted_probs = np.sort(prob)[::-1]
        confidence_gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
        
        # Feature 7: Top-3 confidence values
        top3_conf = sorted_probs[:3]
        top1_conf = top3_conf[0]
        top2_conf = top3_conf[1] if len(top3_conf) > 1 else 0.0
        top3_conf_val = top3_conf[2] if len(top3_conf) > 2 else 0.0
        
        # Feature 8: Number of classes with confidence > 0.1
        high_confidence_count = np.sum(prob > 0.1)
        
        # Feature 9: Loss value (if provided)
        loss_value = loss_values[i] if loss_values is not None else 0.0
        
        # Feature 10: Margin (difference between true class confidence and max other class)
        other_classes = np.delete(prob, true_label)
        max_other_conf = other_classes.max() if len(other_classes) > 0 else 0.0
        margin = true_class_confidence - max_other_conf
        
        feature_vector = [
            max_confidence,
            entropy,
            correctness,
            pred_confidence,
            true_class_confidence,
            confidence_gap,
            top1_conf,
            top2_conf,
            top3_conf_val,
            high_confidence_count,
            loss_value,
            margin
        ]
        
        features.append(feature_vector)
    
    return np.array(features)


def compute_loss_per_sample(
    probabilities: np.ndarray,
    true_labels: np.ndarray
) -> np.ndarray:
    """
    Compute cross-entropy loss per sample.
    
    Args:
        probabilities: (n_samples, n_classes) prediction probabilities
        true_labels: (n_samples,) true class labels
        
    Returns:
        (n_samples,) array of loss values
    """
    n_samples = len(probabilities)
    losses = []
    
    for i in range(n_samples):
        true_label = true_labels[i]
        prob = probabilities[i]
        
        # Cross-entropy loss: -log(p_true_class)
        loss = -np.log(prob[true_label] + 1e-10)
        losses.append(loss)
    
    return np.array(losses)


def create_attack_dataset(
    member_probs: np.ndarray,
    member_preds: np.ndarray,
    member_labels: np.ndarray,
    non_member_probs: np.ndarray,
    non_member_preds: np.ndarray,
    non_member_labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create attack training dataset from shadow model predictions.
    
    This function:
    1. Extracts features from member predictions (label = 1)
    2. Extracts features from non-member predictions (label = 0)
    3. Combines them into a balanced attack dataset
    
    Args:
        member_probs: (n_members, n_classes) prediction probabilities on members
        member_preds: (n_members,) predicted classes on members
        member_labels: (n_members,) true labels for members
        non_member_probs: (n_non_members, n_classes) prediction probabilities on non-members
        non_member_preds: (n_non_members,) predicted classes on non-members
        non_member_labels: (n_non_members,) true labels for non-members
        
    Returns:
        Tuple of (features, labels)
        - features: (n_samples, n_features) array of extracted features
        - labels: (n_samples,) array of membership labels (1=member, 0=non-member)
    """
    # Compute loss values
    member_losses = compute_loss_per_sample(member_probs, member_labels)
    non_member_losses = compute_loss_per_sample(non_member_probs, non_member_labels)
    
    # Extract features
    member_features = extract_prediction_features(
        member_probs,
        member_preds,
        member_labels,
        loss_values=member_losses
    )
    
    non_member_features = extract_prediction_features(
        non_member_probs,
        non_member_preds,
        non_member_labels,
        loss_values=non_member_losses
    )
    
    # Combine features and labels
    features = np.vstack([member_features, non_member_features])
    labels = np.hstack([
        np.ones(len(member_features)),  # 1 = member
        np.zeros(len(non_member_features))  # 0 = non-member
    ])
    
    logger.info(f"Created attack dataset: {len(features)} samples, {len(member_features)} members, {len(non_member_features)} non-members")
    
    return features, labels


def train_attack_model(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    num_epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    device: torch.device = None,
    train_ratio: float = 0.8
) -> Tuple[nn.Module, dict]:
    """
    Train the attack classifier model.
    
    The attack model learns to distinguish members from non-members based on
    prediction features extracted from shadow models.
    
    Args:
        model: Attack model to train (MLP or Logistic Regression)
        features: (n_samples, n_features) feature array
        labels: (n_samples,) membership labels (1=member, 0=non-member)
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to train on
        train_ratio: Ratio of data to use for training (rest for validation)
        
    Returns:
        Tuple of (trained_model, training_history)
        - trained_model: Trained attack classifier
        - training_history: Dictionary with training metrics
    """
    if device is None:
        device = torch.device("cpu")
    
    model.to(device)
    
    # Split into train and validation sets
    n_samples = len(features)
    n_train = int(n_samples * train_ratio)
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    val_features = features[val_indices]
    val_labels = labels[val_indices]
    
    # Convert to tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(train_features),
        torch.LongTensor(train_labels)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_features),
        torch.LongTensor(val_labels)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    logger.info(f"Training attack model for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
    
    model.eval()
    logger.info("Attack model training completed")
    
    return model, history


def evaluate_attack_model(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    device: torch.device = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate attack model and return predictions.
    
    Args:
        model: Trained attack model
        features: (n_samples, n_features) feature array
        labels: (n_samples,) true membership labels
        device: Device to run inference on
        
    Returns:
        Tuple of (predictions, probabilities, true_labels)
        - predictions: (n_samples,) predicted membership (0 or 1)
        - probabilities: (n_samples, 2) prediction probabilities [P(non-member), P(member)]
        - true_labels: (n_samples,) true membership labels
    """
    if device is None:
        device = torch.device("cpu")
    
    model.to(device)
    model.eval()
    
    dataset = TensorDataset(
        torch.FloatTensor(features),
        torch.LongTensor(labels)
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(targets.cpu().numpy())
    
    predictions = np.concatenate(all_preds)
    probabilities = np.vstack(all_probs)
    true_labels = np.concatenate(all_labels)
    
    return predictions, probabilities, true_labels
