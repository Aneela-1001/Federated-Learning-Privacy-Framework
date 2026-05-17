"""
Per-Client Membership Inference Attack (FL-Specific).

This attack is specific to Federated Learning and tries to determine WHICH CLIENT
contributed a particular data sample, rather than just whether it was in the training set.

Attack Goal:
Given a data sample and the global model's prediction output, infer which client
contributed that sample during federated training.

Threat Model:
- Black-box attacker (only has access to model predictions)
- No access to local models or client updates
- Only probability vectors from the global model
- Client inference is treated as a multi-class classification problem

Attack Methodology:
1. Train shadow models - one per client (simulating federated training)
2. Collect predictions from shadow models on their training data
3. Label predictions with corresponding client IDs
4. Train multi-class attack classifier (predictions -> client ID)
5. Evaluate on real federated global model
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import List, Tuple, Dict
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import using importlib
import importlib.util

# Import client module
try:
    client_spec = importlib.util.spec_from_file_location(
        "client_module",
        project_root / "client" / "client.py"
    )
    client_module = importlib.util.module_from_spec(client_spec)
    client_spec.loader.exec_module(client_module)
    FlowerClient = client_module.FlowerClient
    load_data = client_module.load_data
except Exception as e:
    print(f"Warning: Could not import client module: {e}")
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
    print(f"Warning: Could not import shared model: {e}")
    Net = None

# Try to import sklearn for evaluation
try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        classification_report,
        precision_recall_fscore_support
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerClientMIAttack:
    """
    Per-Client Membership Inference Attack.
    
    This attack determines which client contributed a data sample by analyzing
    the global model's predictions on that sample.
    """
    
    def __init__(
        self,
        num_clients: int = 3,
        num_features: int = 784,
        num_classes: int = 10,
        device: torch.device = None
    ):
        """
        Initialize per-client MIA attack.
        
        Args:
            num_clients: Number of clients in federated learning
            num_features: Number of input features (784 for MNIST)
            num_classes: Number of output classes (10 for MNIST)
            device: Device to run on (CPU/GPU)
        """
        self.num_clients = num_clients
        self.num_features = num_features
        self.num_classes = num_classes
        self.device = device if device else torch.device("cpu")
        
        self.shadow_models = {}  # client_id -> shadow_model
        self.attack_classifier = None
        
        logger.info(f"Per-Client MIA initialized for {num_clients} clients")
    
    def train_shadow_model_per_client(
        self,
        client_id: str,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        num_epochs: int = 5
    ) -> nn.Module:
        """
        Train a shadow model on data from a single client.
        
        This simulates what a client's local model would look like after
        federated training. Each shadow model is trained only on one client's data.
        
        Args:
            client_id: Client identifier (e.g., "client-1")
            train_data: Training data for this client
            train_labels: Training labels for this client
            num_epochs: Number of training epochs
            
        Returns:
            Trained shadow model
        """
        logger.info(f"Training shadow model for {client_id}...")
        
        # Create model
        shadow_model = Net(num_features=self.num_features, num_classes=self.num_classes)
        shadow_model.to(self.device)
        shadow_model.train()
        
        # Create data loader
        dataset = TensorDataset(train_data, train_labels)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Train model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(shadow_model.parameters(), lr=0.01, momentum=0.9)
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = shadow_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            if (epoch + 1) % 2 == 0 or epoch == 0:
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
                logger.info(f"  Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        shadow_model.eval()
        logger.info(f"Shadow model for {client_id} trained successfully")
        
        return shadow_model
    
    def extract_prediction_features(
        self,
        probabilities: np.ndarray,
        predicted_classes: np.ndarray,
        true_labels: np.ndarray
    ) -> np.ndarray:
        """
        Extract features from model predictions for client inference.
        
        These features capture patterns that might indicate which client
        contributed the sample (e.g., confidence patterns, entropy, etc.).
        
        Args:
            probabilities: (n_samples, n_classes) prediction probabilities
            predicted_classes: (n_samples,) predicted class indices
            true_labels: (n_samples,) true class labels
            
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
            
            # Feature 2: Prediction entropy
            entropy = -np.sum(prob * np.log(prob + 1e-10))
            
            # Feature 3: Correctness (1 if correct, 0 if wrong)
            correctness = 1.0 if pred_class == true_label else 0.0
            
            # Feature 4: Confidence of predicted class
            pred_confidence = prob[pred_class]
            
            # Feature 5: Confidence of true class
            true_class_confidence = prob[true_label]
            
            # Feature 6: Top-2 confidence gap
            sorted_probs = np.sort(prob)[::-1]
            confidence_gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
            
            # Feature 7-9: Top-3 confidence values
            top3_conf = sorted_probs[:3]
            top1_conf = top3_conf[0]
            top2_conf = top3_conf[1] if len(top3_conf) > 1 else 0.0
            top3_conf_val = top3_conf[2] if len(top3_conf) > 2 else 0.0
            
            # Feature 10: Number of classes with confidence > 0.1
            high_confidence_count = np.sum(prob > 0.1)
            
            # Feature 11: Margin (true class confidence - max other class)
            other_classes = np.delete(prob, true_label)
            max_other_conf = other_classes.max() if len(other_classes) > 0 else 0.0
            margin = true_class_confidence - max_other_conf
            
            # Feature 12: Standard deviation of probabilities
            prob_std = np.std(prob)
            
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
                margin,
                prob_std
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def query_model(
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
                
                # Black-box access: only forward pass
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
    
    def build_attack_dataset(
        self,
        client_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build attack training dataset from shadow model predictions.
        
        For each client:
        1. Train shadow model on that client's data
        2. Query shadow model on its training data
        3. Extract prediction features
        4. Label features with client ID
        
        Args:
            client_data: Dictionary mapping client_id -> (data, labels)
            
        Returns:
            Tuple of (features, client_labels)
            - features: (n_samples, n_features) prediction features
            - client_labels: (n_samples,) client IDs (0, 1, 2, ...)
        """
        logger.info("Building attack dataset from shadow model predictions...")
        
        all_features = []
        all_client_labels = []
        
        # Process each client
        for client_idx, (client_id, (data, labels)) in enumerate(client_data.items()):
            logger.info(f"Processing {client_id} (client {client_idx})...")
            
            # Train shadow model for this client
            shadow_model = self.train_shadow_model_per_client(
                client_id=client_id,
                train_data=data,
                train_labels=labels,
                num_epochs=5
            )
            
            self.shadow_models[client_id] = shadow_model
            
            # Query shadow model on its training data
            dataset = TensorDataset(data, labels)
            data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            probs, preds, true_labels = self.query_model(shadow_model, data_loader)
            
            # Extract features
            features = self.extract_prediction_features(probs, preds, true_labels)
            
            # Label with client ID (0, 1, 2, ...)
            client_labels = np.full(len(features), client_idx)
            
            all_features.append(features)
            all_client_labels.append(client_labels)
            
            logger.info(f"  Collected {len(features)} samples from {client_id}")
        
        # Combine all features and labels
        combined_features = np.vstack(all_features)
        combined_labels = np.concatenate(all_client_labels)
        
        logger.info(f"Built attack dataset: {len(combined_features)} samples from {self.num_clients} clients")
        
        return combined_features, combined_labels
    
    def train_attack_classifier(
        self,
        features: np.ndarray,
        client_labels: np.ndarray,
        num_epochs: int = 50
    ):
        """
        Train multi-class attack classifier.
        
        This classifier learns to predict which client contributed a sample
        based on prediction features extracted from the global model.
        
        Args:
            features: (n_samples, n_features) prediction features
            client_labels: (n_samples,) client IDs
            num_epochs: Number of training epochs
        """
        logger.info("Training multi-class attack classifier...")
        
        # Create model (multi-class classifier)
        input_dim = features.shape[1]
        hidden_dim = 64
        
        classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, self.num_clients)  # Output: num_clients classes
        )
        
        classifier.to(self.device)
        
        # Split into train and validation
        n_samples = len(features)
        n_train = int(n_samples * 0.8)
        
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        train_features = features[train_indices]
        train_labels = client_labels[train_indices]
        val_features = features[val_indices]
        val_labels = client_labels[val_indices]
        
        # Convert to tensors
        train_dataset = TensorDataset(
            torch.FloatTensor(train_features),
            torch.LongTensor(train_labels)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(val_features),
            torch.LongTensor(val_labels)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=0.001)
        
        logger.info(f"Training on {len(train_features)} samples, validating on {len(val_features)} samples...")
        
        for epoch in range(num_epochs):
            # Training
            classifier.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = classifier(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
            
            train_acc = train_correct / train_total
            
            # Validation
            classifier.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    outputs = classifier(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()
            
            val_acc = val_correct / val_total
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"  Epoch [{epoch+1}/{num_epochs}] - "
                    f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
                )
        
        classifier.eval()
        self.attack_classifier = classifier
        
        logger.info(f"Attack classifier trained (final val acc: {val_acc:.4f})")
    
    def attack_global_model(
        self,
        global_model: nn.Module,
        client_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Attack the real federated global model.
        
        Args:
            global_model: Trained federated global model
            client_data: Dictionary mapping client_id -> (data, labels)
            
        Returns:
            Tuple of (predictions, true_client_labels)
        """
        if self.attack_classifier is None:
            raise ValueError("Attack classifier not trained. Call train_attack_classifier first.")
        
        logger.info("Attacking real federated global model...")
        
        global_model.to(self.device)
        global_model.eval()
        
        all_features = []
        all_true_labels = []
        
        # Process each client's data
        for client_idx, (client_id, (data, labels)) in enumerate(client_data.items()):
            logger.info(f"Querying global model on {client_id} data...")
            
            # Query global model (black-box access)
            dataset = TensorDataset(data, labels)
            data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            probs, preds, true_labels = self.query_model(global_model, data_loader)
            
            # Extract features
            features = self.extract_prediction_features(probs, preds, true_labels)
            
            # True client labels
            client_labels = np.full(len(features), client_idx)
            
            all_features.append(features)
            all_true_labels.append(client_labels)
            
            logger.info(f"  Collected {len(features)} samples from {client_id}")
        
        # Combine
        test_features = np.vstack(all_features)
        true_client_labels = np.concatenate(all_true_labels)
        
        # Predict client IDs using attack classifier
        test_dataset = TensorDataset(
            torch.FloatTensor(test_features),
            torch.LongTensor(true_client_labels)
        )
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        all_predictions = []
        
        self.attack_classifier.eval()
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.attack_classifier(inputs)
                _, predicted = torch.max(outputs, 1)
                all_predictions.append(predicted.cpu().numpy())
        
        predictions = np.concatenate(all_predictions)
        
        logger.info(f"Attack completed on {len(test_features)} samples")
        
        return predictions, true_client_labels


def evaluate_per_client_attack(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    num_clients: int,
    client_names: List[str]
):
    """
    Evaluate per-client attack performance.
    
    Args:
        predictions: (n_samples,) predicted client IDs
        true_labels: (n_samples,) true client IDs
        num_clients: Number of clients
        client_names: List of client names for display
    """
    if not SKLEARN_AVAILABLE:
        print("Warning: scikit-learn not available for detailed evaluation")
        return
    
    # Calculate overall accuracy
    accuracy = accuracy_score(true_labels, predictions)
    
    # Random guess baseline
    random_baseline = 1.0 / num_clients
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions, labels=list(range(num_clients)))
    
    # Per-client metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, labels=list(range(num_clients)), zero_division=0
    )
    
    # Print results
    print("\n" + "="*70)
    print("PER-CLIENT MEMBERSHIP INFERENCE ATTACK RESULTS")
    print("="*70)
    
    print(f"\nOverall Client Inference Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Random Guess Baseline: {random_baseline:.4f} ({random_baseline*100:.2f}%)")
    print(f"Attack Improvement over Random: {(accuracy - random_baseline)*100:.2f}%")
    
    if accuracy < random_baseline * 1.1:
        print("\n[EXCELLENT] Attack accuracy near random baseline")
        print("  Privacy mechanisms successfully prevent client identification!")
    elif accuracy < random_baseline * 1.3:
        print("\n[GOOD] Attack accuracy slightly above random")
        print("  Privacy mechanisms provide good protection")
    elif accuracy < random_baseline * 1.5:
        print("\n[MODERATE] Attack accuracy moderately above random")
        print("  Consider strengthening privacy mechanisms")
    else:
        print("\n[WEAK] Attack accuracy significantly above random")
        print("  Privacy mechanisms may need improvement")
    
    print("\n" + "-"*70)
    print("PER-CLIENT PERFORMANCE:")
    print("-"*70)
    print(f"{'Client':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Samples':<10}")
    print("-"*70)
    
    for i in range(num_clients):
        client_name = client_names[i] if i < len(client_names) else f"Client-{i+1}"
        print(f"{client_name:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
    
    print("\n" + "-"*70)
    print("CONFUSION MATRIX:")
    print("-"*70)
    print("Rows = True Client, Columns = Predicted Client")
    print("\n" + " " * 15, end="")
    for i in range(num_clients):
        client_name = client_names[i] if i < len(client_names) else f"C{i+1}"
        print(f"{client_name:>10}", end="")
    print()
    
    for i in range(num_clients):
        client_name = client_names[i] if i < len(client_names) else f"C{i+1}"
        print(f"{client_name:<15}", end="")
        for j in range(num_clients):
            print(f"{cm[i, j]:>10}", end="")
        print()
    
    print("\n" + "="*70)


def main():
    """
    Main function to run per-client membership inference attack.
    """
    print("\n" + "="*70)
    print("PER-CLIENT MEMBERSHIP INFERENCE ATTACK (FL-SPECIFIC)")
    print("="*70)
    print("\nThis attack determines WHICH CLIENT contributed a data sample")
    print("by analyzing the global model's predictions.")
    print("\nThreat Model: Black-box attacker (only model predictions)")
    print("="*70)
    
    if load_data is None or Net is None:
        print("\nERROR: Required modules not available.")
        print("Please ensure client/client.py and shared/model.py exist.")
        return
    
    device = torch.device("cpu")
    num_clients = 5  # Test with 5 clients
    num_features = 784
    num_classes = 10
    
    # Load data for multiple clients
    print("\nLoading data for multiple clients...")
    client_data = {}
    client_names = []
    
    for i in range(1, num_clients + 1):
        client_id = f"client-{i}"
        try:
            trainloader, valloader = load_data(client_id)
            
            # Get training data
            train_data = []
            train_labels = []
            for inputs, labels in trainloader:
                train_data.append(inputs)
                train_labels.append(labels)
            
            train_data = torch.cat(train_data, dim=0)[:500]  # Limit for faster testing
            train_labels = torch.cat(train_labels, dim=0)[:500]
            
            client_data[client_id] = (train_data, train_labels)
            client_names.append(client_id)
            
            print(f"  Loaded {len(train_data)} samples from {client_id}")
        except Exception as e:
            print(f"  Warning: Could not load data for {client_id}: {e}")
            # Create dummy data for testing
            train_data = torch.randn(500, num_features)
            train_labels = torch.randint(0, num_classes, (500,))
            client_data[client_id] = (train_data, train_labels)
            client_names.append(client_id)
            print(f"  Using dummy data for {client_id}")
    
    if len(client_data) == 0:
        print("ERROR: No client data available")
        return
    
    num_clients = len(client_data)
    print(f"\nLoaded data from {num_clients} clients")
    
    # Initialize attack
    print("\nInitializing per-client MIA attack...")
    attack = PerClientMIAttack(
        num_clients=num_clients,
        num_features=num_features,
        num_classes=num_classes,
        device=device
    )
    
    # Step 1: Build attack dataset from shadow models
    print("\n" + "="*70)
    print("STEP 1: Building Attack Dataset from Shadow Models")
    print("="*70)
    attack_features, attack_labels = attack.build_attack_dataset(client_data)
    
    # Step 2: Train attack classifier
    print("\n" + "="*70)
    print("STEP 2: Training Multi-Class Attack Classifier")
    print("="*70)
    attack.train_attack_classifier(
        features=attack_features,
        client_labels=attack_labels,
        num_epochs=30  # Reduced for faster testing
    )
    
    # Step 3: Train a simulated federated global model
    print("\n" + "="*70)
    print("STEP 3: Training Simulated Federated Global Model")
    print("="*70)
    print("(This simulates the real federated learning scenario)")
    
    global_model = Net(num_features=num_features, num_classes=num_classes)
    global_model.to(device)
    
    # Train global model on combined data from all clients
    all_global_data = []
    all_global_labels = []
    for data, labels in client_data.values():
        all_global_data.append(data)
        all_global_labels.append(labels)
    
    global_train_data = torch.cat(all_global_data, dim=0)
    global_train_labels = torch.cat(all_global_labels, dim=0)
    
    global_dataset = TensorDataset(global_train_data, global_train_labels)
    global_loader = DataLoader(global_dataset, batch_size=32, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(global_model.parameters(), lr=0.01)
    global_model.train()
    
    print("Training global model...")
    for epoch in range(5):
        for inputs, labels in global_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = global_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    global_model.eval()
    print("Global model trained")
    
    # Step 4: Attack the global model
    print("\n" + "="*70)
    print("STEP 4: Attacking Real Federated Global Model")
    print("="*70)
    predictions, true_labels = attack.attack_global_model(global_model, client_data)
    
    # Step 5: Evaluate attack
    print("\n" + "="*70)
    print("STEP 5: Evaluating Attack Performance")
    print("="*70)
    evaluate_per_client_attack(
        predictions=predictions,
        true_labels=true_labels,
        num_clients=num_clients,
        client_names=client_names
    )
    
    print("\n" + "="*70)
    print("ATTACK COMPLETE")
    print("="*70)
    print("\nInterpretation:")
    print("- Lower accuracy (closer to random baseline) = Better privacy protection")
    print("- Higher accuracy = Attack can identify which client contributed samples")
    print("- Random baseline = 1 / num_clients (e.g., 33.33% for 3 clients)")
    print("="*70)


if __name__ == "__main__":
    main()
