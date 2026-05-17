"""
Black-Box Membership Inference Attack (MIA) Implementation.

This is the main module that orchestrates the complete black-box MIA attack pipeline:

1. Shadow Model Training: Train shadow models on auxiliary data to mimic target model behavior
2. Feature Extraction: Extract prediction features from shadow models
3. Attack Model Training: Train a binary classifier to distinguish members from non-members
4. Target Model Evaluation: Apply the attack to the target model (black-box access only)
5. Performance Evaluation: Measure attack accuracy, precision, recall, and AUC

The attack assumes:
- Black-box access to target model (only predictions, no weights/gradients)
- Auxiliary data available for shadow model training
- MNIST-like classification task

Usage:
    python blackbox_mia.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional, List
import logging
from pathlib import Path
import sys

# Import project modules
from shadow_model import (
    ShadowModel,
    train_shadow_model,
    create_shadow_datasets,
    generate_shadow_predictions,
    train_multiple_shadow_models
)
from attack_model import (
    AttackMLP,
    AttackLogisticRegression,
    extract_prediction_features,
    compute_loss_per_sample,
    create_attack_dataset,
    train_attack_model,
    evaluate_attack_model
)

# Import client data loading (assuming it exists)
try:
    import importlib.util
    project_root = Path(__file__).parent
    client_spec = importlib.util.spec_from_file_location(
        "client_module",
        project_root / "client" / "client.py"
    )
    client_module = importlib.util.module_from_spec(client_spec)
    client_spec.loader.exec_module(client_module)
    load_data = client_module.load_data
except Exception as e:
    logging.warning(f"Could not import load_data: {e}")
    load_data = None

# Import shared model
try:
    shared_model_spec = importlib.util.spec_from_file_location(
        "shared_model",
        project_root / "shared" / "model.py"
    )
    shared_model = importlib.util.module_from_spec(shared_model_spec)
    shared_model_spec.loader.exec_module(shared_model)
    Net = shared_model.Net
    get_model_parameters = shared_model.get_model_parameters
    set_model_parameters = shared_model.set_model_parameters
except Exception as e:
    logging.warning(f"Could not import shared model: {e}")
    Net = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BlackBoxMIA:
    """
    Main class for black-box membership inference attack.
    
    This class orchestrates the complete attack pipeline:
    1. Train shadow models on auxiliary data
    2. Extract prediction features from shadow models
    3. Train attack classifier
    4. Evaluate attack on target model
    """
    
    def __init__(
        self,
        num_features: int = 784,
        num_classes: int = 10,
        device: torch.device = None
    ):
        """
        Initialize black-box MIA attack.
        
        Args:
            num_features: Number of input features (784 for MNIST)
            num_classes: Number of output classes (10 for MNIST)
            device: Device to run on (CPU/GPU)
        """
        self.num_features = num_features
        self.num_classes = num_classes
        self.device = device if device else torch.device("cpu")
        
        self.shadow_models = []
        self.attack_model = None
        self.attack_model_type = None
        
        logger.info(f"Black-box MIA initialized (features={num_features}, classes={num_classes})")
    
    def prepare_auxiliary_data(
        self,
        member_data: torch.Tensor,
        member_labels: torch.Tensor,
        non_member_data: torch.Tensor,
        non_member_labels: torch.Tensor
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare auxiliary data for shadow model training.
        
        In black-box MIA, the attacker doesn't have access to the target model's
        training data. Instead, they use auxiliary data that:
        - Has similar distribution to target model's training data
        - Is split into member (for shadow training) and non-member (holdout) sets
        
        Args:
            member_data: Data to train shadow models on
            member_labels: Labels for member data
            non_member_data: Holdout data (not used for shadow training)
            non_member_labels: Labels for non-member data
            
        Returns:
            Tuple of (member_loader, non_member_loader)
        """
        member_loader, non_member_loader = create_shadow_datasets(
            member_data,
            member_labels,
            non_member_data,
            non_member_labels,
            batch_size=32
        )
        
        logger.info(f"Prepared auxiliary data: {len(member_data)} members, {len(non_member_data)} non-members")
        
        return member_loader, non_member_loader
    
    def train_shadow_models(
        self,
        member_data: torch.Tensor,
        member_labels: torch.Tensor,
        non_member_data: torch.Tensor,
        non_member_labels: torch.Tensor,
        num_shadow_models: int = 3,
        num_epochs: int = 10
    ):
        """
        Train shadow models on auxiliary data.
        
        Shadow models help the attacker understand:
        - How models behave on training data (members) vs. non-training data (non-members)
        - What prediction patterns indicate membership
        
        Args:
            member_data: Training data for shadow models
            member_labels: Training labels
            non_member_data: Holdout data
            non_member_labels: Holdout labels
            num_shadow_models: Number of shadow models to train
            num_epochs: Number of epochs per shadow model
        """
        logger.info(f"Training {num_shadow_models} shadow models...")
        
        shadow_results = train_multiple_shadow_models(
            num_shadow_models=num_shadow_models,
            member_data=member_data,
            member_labels=member_labels,
            non_member_data=non_member_data,
            non_member_labels=non_member_labels,
            num_features=self.num_features,
            num_classes=self.num_classes,
            num_epochs=num_epochs,
            batch_size=32,
            device=self.device
        )
        
        self.shadow_models = shadow_results
        logger.info(f"Successfully trained {len(self.shadow_models)} shadow models")
    
    def build_attack_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build attack training dataset from shadow model predictions.
        
        This function:
        1. Queries shadow models on member and non-member data (black-box access)
        2. Extracts prediction features
        3. Creates labeled dataset (1=member, 0=non-member)
        
        Returns:
            Tuple of (features, labels) for attack model training
        """
        logger.info("Building attack dataset from shadow model predictions...")
        
        all_member_features = []
        all_non_member_features = []
        
        # Process each shadow model's predictions
        for shadow_result in self.shadow_models:
            if len(shadow_result) == 7:
                # New format with labels
                shadow_model, member_probs, member_preds, member_labels, non_member_probs, non_member_preds, non_member_labels = shadow_result
            else:
                # Old format without labels (backward compatibility)
                shadow_model, member_probs, member_preds, non_member_probs, non_member_preds = shadow_result
                n_member = len(member_probs)
                n_non_member = len(non_member_probs)
                member_labels = np.random.randint(0, self.num_classes, size=n_member)
                non_member_labels = np.random.randint(0, self.num_classes, size=n_non_member)
            
            # Compute losses
            member_losses = compute_loss_per_sample(member_probs, member_labels)
            non_member_losses = compute_loss_per_sample(non_member_probs, non_member_labels)
            
            # Extract features from this shadow model's predictions
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
            
            all_member_features.append(member_features)
            all_non_member_features.append(non_member_features)
        
        # Concatenate features from all shadow models (creates larger, more diverse dataset)
        combined_member_features = np.vstack(all_member_features)
        combined_non_member_features = np.vstack(all_non_member_features)
        
        # Create labels (1 = member, 0 = non-member)
        member_labels = np.ones(len(combined_member_features))
        non_member_labels = np.zeros(len(combined_non_member_features))
        
        # Combine features and labels
        features = np.vstack([combined_member_features, combined_non_member_features])
        labels = np.hstack([member_labels, non_member_labels])
        
        logger.info(f"Built attack dataset: {len(features)} samples ({len(combined_member_features)} members, {len(combined_non_member_features)} non-members)")
        
        return features, labels
    
    def train_attack_classifier(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        attack_model_type: str = "mlp",
        num_epochs: int = 50
    ):
        """
        Train attack classifier to distinguish members from non-members.
        
        The attack classifier learns patterns in prediction features that indicate
        whether a sample was in the training set (member) or not (non-member).
        
        Args:
            features: (n_samples, n_features) feature array
            labels: (n_samples,) membership labels (1=member, 0=non-member)
            attack_model_type: Type of attack model ("mlp" or "logistic")
            num_epochs: Number of training epochs
        """
        logger.info(f"Training attack classifier ({attack_model_type})...")
        
        # Create attack model
        input_dim = features.shape[1]
        
        if attack_model_type == "mlp":
            model = AttackMLP(input_dim=input_dim, hidden_dim=64, num_classes=2)
        elif attack_model_type == "logistic":
            model = AttackLogisticRegression(input_dim=input_dim, num_classes=2)
        else:
            raise ValueError(f"Unknown attack model type: {attack_model_type}")
        
        # Train model
        trained_model, history = train_attack_model(
            model,
            features,
            labels,
            num_epochs=num_epochs,
            batch_size=64,
            learning_rate=0.001,
            device=self.device
        )
        
        self.attack_model = trained_model
        self.attack_model_type = attack_model_type
        
        logger.info(f"Attack classifier trained (final val acc: {history['val_acc'][-1]:.4f})")
    
    def attack_target_model(
        self,
        target_model: nn.Module,
        member_data: torch.Tensor,
        member_labels: torch.Tensor,
        non_member_data: torch.Tensor,
        non_member_labels: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply attack to target model (black-box access only).
        
        This function:
        1. Queries target model on member and non-member data (black-box)
        2. Extracts prediction features
        3. Uses trained attack classifier to predict membership
        
        Args:
            target_model: Target model to attack (black-box access only)
            member_data: Data that was in target model's training set
            member_labels: Labels for member data
            non_member_data: Data NOT in target model's training set
            non_member_labels: Labels for non-member data
            
        Returns:
            Tuple of (predictions, probabilities, true_labels)
        """
        if self.attack_model is None:
            raise ValueError("Attack model not trained. Call train_attack_classifier first.")
        
        logger.info("Attacking target model (black-box access only)...")
        
        target_model.to(self.device)
        target_model.eval()
        
        # Create data loaders
        member_dataset = TensorDataset(member_data, member_labels)
        non_member_dataset = TensorDataset(non_member_data, non_member_labels)
        
        member_loader = DataLoader(member_dataset, batch_size=32, shuffle=False)
        non_member_loader = DataLoader(non_member_dataset, batch_size=32, shuffle=False)
        
        # Query target model (black-box access - only predictions)
        member_probs, member_preds, _ = self._query_model(target_model, member_loader)
        non_member_probs, non_member_preds, _ = self._query_model(target_model, non_member_loader)
        
        # Compute losses
        member_losses = compute_loss_per_sample(member_probs, member_labels.numpy())
        non_member_losses = compute_loss_per_sample(non_member_probs, non_member_labels.numpy())
        
        # Extract features
        member_features = extract_prediction_features(
            member_probs,
            member_preds,
            member_labels.numpy(),
            loss_values=member_losses
        )
        
        non_member_features = extract_prediction_features(
            non_member_probs,
            non_member_preds,
            non_member_labels.numpy(),
            loss_values=non_member_losses
        )
        
        # Combine features and labels
        test_features = np.vstack([member_features, non_member_features])
        test_labels = np.hstack([
            np.ones(len(member_features)),  # 1 = member
            np.zeros(len(non_member_features))  # 0 = non-member
        ])
        
        # Predict membership using attack classifier
        predictions, probabilities, _ = evaluate_attack_model(
            self.attack_model,
            test_features,
            test_labels,
            device=self.device
        )
        
        logger.info(f"Attack completed on {len(test_features)} samples")
        
        return predictions, probabilities, test_labels
    
    def _query_model(
        self,
        model: nn.Module,
        data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Query model in black-box fashion (only predictions, no internals).
        
        Args:
            model: Model to query
            data_loader: DataLoader with data to query
            
        Returns:
            Tuple of (probabilities, predicted_classes, true_labels)
        """
        model.eval()
        all_probs = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                
                # Black-box access: only forward pass, no gradients/weights
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.numpy())
        
        probabilities = np.vstack(all_probs)
        predicted_classes = np.concatenate(all_preds)
        true_labels = np.concatenate(all_labels)
        
        return probabilities, predicted_classes, true_labels


def evaluate_attack_performance(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    true_labels: np.ndarray
) -> dict:
    """
    Evaluate attack performance using standard metrics.
    
    Args:
        predictions: (n_samples,) predicted membership (0 or 1)
        probabilities: (n_samples, 2) prediction probabilities
        true_labels: (n_samples,) true membership labels (1=member, 0=non-member)
        
    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix
    )
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    try:
        auc = roc_auc_score(true_labels, probabilities[:, 1])
    except:
        auc = 0.0
    
    cm = confusion_matrix(true_labels, predictions)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc,
        'confusion_matrix': cm
    }
    
    return metrics


def print_attack_results(metrics: dict):
    """
    Print attack evaluation results in a readable format.
    
    Args:
        metrics: Dictionary with evaluation metrics
    """
    print("\n" + "="*70)
    print("BLACK-BOX MEMBERSHIP INFERENCE ATTACK RESULTS")
    print("="*70)
    print(f"Attack Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  (Random guessing: 50.00%)")
    print(f"Precision:         {metrics['precision']:.4f}")
    print(f"Recall:            {metrics['recall']:.4f}")
    print(f"F1-Score:          {metrics['f1_score']:.4f}")
    print(f"AUC-ROC:           {metrics['auc_roc']:.4f}")
    print(f"  (AUC > 0.5 indicates attack success)")
    
    print("\nConfusion Matrix:")
    print(f"                  Predicted")
    print(f"                 Non-Member  Member")
    print(f"Actual Non-Member    {metrics['confusion_matrix'][0,0]:4d}      {metrics['confusion_matrix'][0,1]:4d}")
    print(f"        Member        {metrics['confusion_matrix'][1,0]:4d}      {metrics['confusion_matrix'][1,1]:4d}")
    
    print("\n" + "-"*70)
    print("PRIVACY EVALUATION:")
    print("-"*70)
    
    acc = metrics['accuracy']
    if acc < 0.55:
        print("[EXCELLENT] Attack accuracy near random (50%)")
        print("  Privacy mechanisms provide strong protection!")
    elif acc < 0.60:
        print("[GOOD] Attack accuracy slightly above random")
        print("  Privacy mechanisms provide good protection")
    elif acc < 0.65:
        print("[MODERATE] Attack accuracy moderately above random")
        print("  Consider strengthening privacy mechanisms")
    else:
        print("[WEAK] Attack accuracy significantly above random")
        print("  Privacy mechanisms may not be sufficient")
        print("  Recommendation: Increase noise_multiplier to 2.0-3.0")
    
    print("="*70)


def main():
    """
    Main function to run complete black-box MIA attack pipeline.
    
    This demonstrates the complete attack:
    1. Load auxiliary data
    2. Train shadow models
    3. Build attack dataset
    4. Train attack classifier
    5. Attack target model
    6. Evaluate results
    """
    print("\n" + "="*70)
    print("BLACK-BOX MEMBERSHIP INFERENCE ATTACK (MIA)")
    print("="*70)
    print("\nThis attack evaluates privacy leakage in federated learning models.")
    print("The attacker has only black-box access (predictions only).")
    print("="*70)
    
    # Check if data loading is available
    if load_data is None:
        print("\nERROR: Could not import load_data function.")
        print("Please ensure client/client.py is available.")
        return
    
    if Net is None:
        print("\nERROR: Could not import model architecture.")
        print("Please ensure shared/model.py is available.")
        return
    
    device = torch.device("cpu")
    
    try:
        # Load data (using client-1 as auxiliary data source)
        print("\nLoading auxiliary data for shadow model training...")
        trainloader, valloader = load_data("client-1")
        
        # Prepare data
        member_data = []
        member_labels = []
        for inputs, labels in trainloader:
            member_data.append(inputs)
            member_labels.append(labels)
        
        member_data = torch.cat(member_data, dim=0)[:1000]  # Limit for faster testing
        member_labels = torch.cat(member_labels, dim=0)[:1000]
        
        non_member_data = []
        non_member_labels = []
        for inputs, labels in valloader:
            non_member_data.append(inputs)
            non_member_labels.append(labels)
        
        non_member_data = torch.cat(non_member_data, dim=0)[:1000]
        non_member_labels = torch.cat(non_member_labels, dim=0)[:1000]
        
        print(f"Loaded data: {len(member_data)} members, {len(non_member_data)} non-members")
        
        # Initialize black-box MIA
        mia = BlackBoxMIA(num_features=784, num_classes=10, device=device)
        
        # Train shadow models
        mia.train_shadow_models(
            member_data=member_data,
            member_labels=member_labels,
            non_member_data=non_member_data,
            non_member_labels=non_member_labels,
            num_shadow_models=3,
            num_epochs=5  # Reduced for faster testing
        )
        
        # Build attack dataset
        attack_features, attack_labels = mia.build_attack_dataset()
        
        # Train attack classifier
        mia.train_attack_classifier(
            features=attack_features,
            labels=attack_labels,
            attack_model_type="mlp",  # or "logistic"
            num_epochs=30  # Reduced for faster testing
        )
        
        # Train target model (simulating federated learning model)
        print("\nTraining target model (simulating federated learning)...")
        target_model = Net(num_features=784, num_classes=10)
        target_model.to(device)
        
        # Train on a subset of member data (simulating target model's training)
        target_train_data = member_data[:500]
        target_train_labels = member_labels[:500]
        target_train_dataset = TensorDataset(target_train_data, target_train_labels)
        target_train_loader = DataLoader(target_train_dataset, batch_size=32, shuffle=True)
        
        import torch.optim as optim
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(target_model.parameters(), lr=0.01)
        
        target_model.train()
        for epoch in range(5):
            for inputs, labels in target_train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = target_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        target_model.eval()
        print("Target model trained")
        
        # Prepare test data (member = training data, non-member = holdout)
        test_member_data = target_train_data[:200]
        test_member_labels = target_train_labels[:200]
        test_non_member_data = non_member_data[:200]
        test_non_member_labels = non_member_labels[:200]
        
        # Attack target model
        predictions, probabilities, true_labels = mia.attack_target_model(
            target_model=target_model,
            member_data=test_member_data,
            member_labels=test_member_labels,
            non_member_data=test_non_member_data,
            non_member_labels=test_non_member_labels
        )
        
        # Evaluate attack
        metrics = evaluate_attack_performance(predictions, probabilities, true_labels)
        print_attack_results(metrics)
        
    except Exception as e:
        logger.error(f"Error during attack: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
