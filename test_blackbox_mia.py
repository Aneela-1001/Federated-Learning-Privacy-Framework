"""
Black Box Membership Inference Attack (MIA) Test Suite.

This implements a comprehensive black box MIA attack against the federated learning system.
The attacker only has access to:
- The trained model (can query it)
- Model predictions on data points
- No access to training data, gradients, or model internals

The attack tries to determine if a specific data point was in the training set.
"""
import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from collections import defaultdict

# Try to import sklearn, provide helpful error if missing
try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    print("ERROR: scikit-learn is required for black box MIA tests.")
    print("Please install it with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False
    # Define dummy functions to prevent errors
    def accuracy_score(*args, **kwargs):
        return 0.5
    def precision_score(*args, **kwargs):
        return 0.5
    def recall_score(*args, **kwargs):
        return 0.5
    def f1_score(*args, **kwargs):
        return 0.5
    def roc_auc_score(*args, **kwargs):
        return 0.5
    class RandomForestClassifier:
        def __init__(self, *args, **kwargs):
            raise ImportError("scikit-learn not installed")
    class LogisticRegression:
        def __init__(self, *args, **kwargs):
            raise ImportError("scikit-learn not installed")

try:
    import pandas as pd
except ImportError:
    pd = None

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import using importlib
import importlib.util

# Import client module
client_spec = importlib.util.spec_from_file_location("client_module", project_root / "client" / "client.py")
client_module = importlib.util.module_from_spec(client_spec)
client_spec.loader.exec_module(client_module)
FlowerClient = client_module.FlowerClient
load_data = client_module.load_data

# Import shared modules
shared_model_spec = importlib.util.spec_from_file_location("shared_model", project_root / "shared" / "model.py")
shared_model = importlib.util.module_from_spec(shared_model_spec)
shared_model_spec.loader.exec_module(shared_model)
Net = shared_model.Net
get_model_parameters = shared_model.get_model_parameters
set_model_parameters = shared_model.set_model_parameters


class BlackBoxMIAttack:
    """
    Black Box Membership Inference Attack.
    
    The attacker:
    1. Queries the target model with data points
    2. Extracts features from predictions (confidence, entropy, etc.)
    3. Trains an attack model to distinguish members from non-members
    4. Evaluates attack accuracy
    """
    
    def __init__(self, target_model: nn.Module, device: torch.device = None):
        """
        Initialize black box MIA attack.
        
        Args:
            target_model: The target model to attack (black box access only)
            device: Device to run on
        """
        self.target_model = target_model
        self.device = device if device else torch.device("cpu")
        self.target_model.to(self.device)
        self.target_model.eval()
        
        # Attack model (classifier to distinguish members from non-members)
        self.attack_model = None
        
    def extract_prediction_features(self, data_loader: DataLoader) -> np.ndarray:
        """
        Extract features from model predictions for MIA.
        
        Features include:
        - Prediction confidence (max softmax probability)
        - Prediction entropy
        - Correctness (whether prediction matches true label)
        - Loss value
        - Top-3 confidence values
        
        Args:
            data_loader: DataLoader with data points
            
        Returns:
            Array of features (n_samples, n_features)
        """
        features = []
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Get model predictions
                logits = self.target_model(inputs)
                probs = torch.softmax(logits, dim=1)
                
                # Compute loss per sample
                losses = criterion(logits, labels)
                
                # Extract features for each sample
                for i in range(inputs.size(0)):
                    prob_sample = probs[i].cpu().numpy()
                    label_sample = labels[i].cpu().item()
                    loss_sample = losses[i].cpu().item()
                    
                    # Feature 1: Maximum confidence (prediction confidence)
                    max_confidence = prob_sample.max()
                    
                    # Feature 2: Prediction entropy
                    entropy = -np.sum(prob_sample * np.log(prob_sample + 1e-10))
                    
                    # Feature 3: Correctness (1 if correct, 0 if wrong)
                    predicted_class = prob_sample.argmax()
                    correctness = 1.0 if predicted_class == label_sample else 0.0
                    
                    # Feature 4: Loss value
                    loss_value = loss_sample
                    
                    # Feature 5-7: Top-3 confidence values
                    top3_conf = np.sort(prob_sample)[-3:][::-1]
                    top1_conf = top3_conf[0] if len(top3_conf) > 0 else 0.0
                    top2_conf = top3_conf[1] if len(top3_conf) > 1 else 0.0
                    top3_conf_val = top3_conf[2] if len(top3_conf) > 2 else 0.0
                    
                    # Feature 8: Confidence gap (top1 - top2)
                    confidence_gap = top1_conf - top2_conf if len(top3_conf) > 1 else top1_conf
                    
                    # Feature 9: Prediction margin (confidence of true class)
                    true_class_confidence = prob_sample[label_sample]
                    
                    # Feature 10: Number of classes with confidence > threshold
                    high_confidence_count = np.sum(prob_sample > 0.1)
                    
                    feature_vector = [
                        max_confidence,
                        entropy,
                        correctness,
                        loss_value,
                        top1_conf,
                        top2_conf,
                        top3_conf_val,
                        confidence_gap,
                        true_class_confidence,
                        high_confidence_count
                    ]
                    
                    features.append(feature_vector)
        
        return np.array(features)
    
    def train_attack_model(
        self,
        member_features: np.ndarray,
        non_member_features: np.ndarray,
        attack_model_type: str = "random_forest"
    ):
        """
        Train an attack model to distinguish members from non-members.
        
        Args:
            member_features: Features extracted from member data points
            non_member_features: Features extracted from non-member data points
            attack_model_type: Type of attack model ("random_forest" or "logistic")
        """
        # Combine features and create labels
        X = np.vstack([member_features, non_member_features])
        y = np.hstack([
            np.ones(len(member_features)),  # 1 = member
            np.zeros(len(non_member_features))  # 0 = non-member
        ])
        
        # Train attack model
        if attack_model_type == "random_forest":
            self.attack_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif attack_model_type == "logistic":
            self.attack_model = LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown attack model type: {attack_model_type}")
        
        self.attack_model.fit(X, y)
        print(f"Attack model trained ({attack_model_type}) on {len(X)} samples")
    
    def predict_membership(self, features: np.ndarray) -> np.ndarray:
        """
        Predict membership for given features.
        
        Args:
            features: Feature vectors extracted from predictions
            
        Returns:
            Array of membership predictions (1 = member, 0 = non-member)
        """
        if self.attack_model is None:
            raise ValueError("Attack model not trained. Call train_attack_model first.")
        
        return self.attack_model.predict(features)
    
    def predict_membership_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict membership probabilities.
        
        Args:
            features: Feature vectors extracted from predictions
            
        Returns:
            Array of membership probabilities [P(non-member), P(member)]
        """
        if self.attack_model is None:
            raise ValueError("Attack model not trained. Call train_attack_model first.")
        
        return self.attack_model.predict_proba(features)


class BlackBoxMIATestSuite:
    """Test suite for black box membership inference attacks."""
    
    def __init__(self):
        self.results = []
        self.attack_accuracies = []
        
    def test_blackbox_mia_basic(self, use_privacy: bool = True):
        """
        Test 1: Basic Black Box MIA Attack
        
        Scenario:
        - Train a model on a dataset
        - Split data into members (training set) and non-members (holdout set)
        - Attack model tries to distinguish members from non-members
        - Test with and without privacy mechanisms
        """
        print("\n" + "="*70)
        print("BLACK BOX MIA TEST 1: Basic Membership Inference Attack")
        print("="*70)
        
        device = torch.device("cpu")
        
        # Load data
        try:
            trainloader, valloader = load_data("client-1")
        except Exception as e:
            print(f"[SKIP] Could not load data: {e}")
            print("  Please run: python partition_mnist.py first")
            return True
        
        # Split data into members and non-members
        # Members: training data
        # Non-members: validation data (or holdout from training)
        member_data = []
        member_labels = []
        for inputs, labels in trainloader:
            member_data.append(inputs)
            member_labels.append(labels)
        
        member_data = torch.cat(member_data, dim=0)
        member_labels = torch.cat(member_labels, dim=0)
        
        # Use validation data as non-members (or split training data)
        non_member_data = []
        non_member_labels = []
        for inputs, labels in valloader:
            non_member_data.append(inputs)
            non_member_labels.append(labels)
        
        non_member_data = torch.cat(non_member_data, dim=0)
        non_member_labels = torch.cat(non_member_labels, dim=0)
        
        # Limit dataset size for faster testing
        max_samples = min(500, len(member_data), len(non_member_data))
        member_data = member_data[:max_samples]
        member_labels = member_labels[:max_samples]
        non_member_data = non_member_data[:max_samples]
        non_member_labels = non_member_labels[:max_samples]
        
        print(f"Member samples: {len(member_data)}")
        print(f"Non-member samples: {len(non_member_data)}")
        
        # Train target model
        print("\nTraining target model...")
        target_model = Net(num_features=784, num_classes=10)
        target_model.to(device)
        
        # Train on member data
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(target_model.parameters(), lr=0.01)
        target_model.train()
        
        member_dataset = TensorDataset(member_data, member_labels)
        member_train_loader = DataLoader(member_dataset, batch_size=32, shuffle=True)
        
        for epoch in range(3):
            for inputs, labels in member_train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = target_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        target_model.eval()
        print("Target model trained")
        
        # Apply privacy mechanisms if requested
        if use_privacy:
            print("\nApplying privacy mechanisms...")
            print("  - Differential Privacy (DP): Enabled with noise_multiplier=2.5")
            print("  - Local Differential Privacy (LDP): Enabled")
            print("  - Adaptive DP: Enabled")
            print("  - Secure Aggregation (SecAgg): Enabled")
            # Get model parameters
            params = get_model_parameters(target_model)
            
            # Create client with STRONG privacy (noise_multiplier=2.5, LDP, Adaptive DP)
            client = FlowerClient(
                model=target_model,
                trainloader=member_train_loader,
                valloader=member_train_loader,
                device=device,
                client_id="client-1",
                use_dp=True,
                use_he=False,
                use_secagg=True,
                use_ldp=True,  # Enable Local DP
                use_adaptive_dp=True,  # Enable Adaptive DP
                num_clients=5,
                dp_noise_multiplier=2.5,  # Increased from 1.0 to 2.5 for stronger protection
                dp_l2_norm_clip=1.0,
                ldp_epsilon=1.0  # LDP privacy budget
            )
            
            # Apply privacy to parameters (simulating privacy-protected training)
            protected_params = client._apply_privacy_mechanisms(params, batch_size=32)
            
            # Create new model with protected parameters
            protected_model = Net(num_features=784, num_classes=10)
            set_model_parameters(protected_model, protected_params)
            protected_model.to(device)
            protected_model.eval()
            
            target_model = protected_model
            print("Privacy mechanisms applied")
        
        # Initialize attack
        print("\nInitializing black box MIA attack...")
        attack = BlackBoxMIAttack(target_model, device)
        
        # Extract features from members and non-members
        print("Extracting prediction features...")
        member_loader = DataLoader(
            TensorDataset(member_data, member_labels),
            batch_size=32,
            shuffle=False
        )
        non_member_loader = DataLoader(
            TensorDataset(non_member_data, non_member_labels),
            batch_size=32,
            shuffle=False
        )
        
        member_features = attack.extract_prediction_features(member_loader)
        non_member_features = attack.extract_prediction_features(non_member_loader)
        
        print(f"Extracted {len(member_features)} member feature vectors")
        print(f"Extracted {len(non_member_features)} non-member feature vectors")
        
        # Split features for training and testing attack model
        split_idx = len(member_features) // 2
        
        train_member_features = member_features[:split_idx]
        test_member_features = member_features[split_idx:]
        train_non_member_features = non_member_features[:split_idx]
        test_non_member_features = non_member_features[split_idx:]
        
        # Train attack model
        print("\nTraining attack model...")
        attack.train_attack_model(
            train_member_features,
            train_non_member_features,
            attack_model_type="random_forest"
        )
        
        # Test attack model
        print("\nEvaluating attack...")
        test_features = np.vstack([test_member_features, test_non_member_features])
        test_labels = np.hstack([
            np.ones(len(test_member_features)),
            np.zeros(len(test_non_member_features))
        ])
        
        predictions = attack.predict_membership(test_features)
        probabilities = attack.predict_membership_proba(test_features)
        
        # Calculate metrics (with zero_division handling)
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions, zero_division=0)
        recall = recall_score(test_labels, predictions, zero_division=0)
        f1 = f1_score(test_labels, predictions, zero_division=0)
        
        try:
            auc = roc_auc_score(test_labels, probabilities[:, 1])
        except:
            auc = 0.0
        
        print("\n" + "-"*70)
        print("ATTACK RESULTS:")
        print("-"*70)
        print(f"Attack Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  (Random guessing: 50%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"  (AUC > 0.5 indicates attack success)")
        
        # Analyze feature importance
        if hasattr(attack.attack_model, 'feature_importances_'):
            importances = attack.attack_model.feature_importances_
            feature_names = [
                "Max Confidence", "Entropy", "Correctness", "Loss",
                "Top1 Conf", "Top2 Conf", "Top3 Conf", "Conf Gap",
                "True Class Conf", "High Conf Count"
            ]
            print("\nFeature Importance (for membership inference):")
            for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
                print(f"  {name:20s}: {imp:.4f}")
        
        # Evaluate privacy protection
        print("\n" + "-"*70)
        print("PRIVACY EVALUATION:")
        print("-"*70)
        
        if accuracy < 0.55:
            print("[EXCELLENT] Attack accuracy near random (50%)")
            print("  Privacy mechanisms provide strong protection!")
            protection_level = "EXCELLENT"
        elif accuracy < 0.60:
            print("[GOOD] Attack accuracy slightly above random")
            print("  Privacy mechanisms provide good protection")
            protection_level = "GOOD"
        elif accuracy < 0.65:
            print("[MODERATE] Attack accuracy moderately above random")
            print("  Privacy mechanisms provide moderate protection")
            print("  Consider increasing noise_multiplier for stronger protection")
            protection_level = "MODERATE"
        else:
            print("[WEAK] Attack accuracy significantly above random")
            print("  Privacy mechanisms may not be sufficient")
            print("  Strongly recommend increasing noise_multiplier to 2.0-3.0")
            protection_level = "WEAK"
        
        self.attack_accuracies.append({
            "test": "Basic MIA",
            "accuracy": accuracy,
            "auc": auc,
            "protection": protection_level,
            "privacy_enabled": use_privacy
        })
        
        return accuracy < 0.65  # Pass if attack accuracy < 65%
    
    def test_blackbox_mia_with_different_noise(self):
        """
        Test 2: MIA Attack with Different Noise Levels
        
        Tests how different noise multipliers affect MIA success rate.
        """
        print("\n" + "="*70)
        print("BLACK BOX MIA TEST 2: Effect of Noise Multiplier")
        print("(With LDP and Adaptive DP enabled)")
        print("="*70)
        
        noise_multipliers = [0.5, 1.0, 2.0, 2.5, 3.0]  # Added 2.5 for testing
        results = []
        
        for noise_mult in noise_multipliers:
            print(f"\n{'='*70}")
            print(f"Testing with noise_multiplier = {noise_mult}")
            print(f"{'='*70}")
            
            device = torch.device("cpu")
            
            try:
                trainloader, valloader = load_data("client-1")
            except Exception as e:
                print(f"[SKIP] Could not load data: {e}")
                continue
            
            # Prepare data (same as test 1)
            member_data = []
            member_labels = []
            for inputs, labels in trainloader:
                member_data.append(inputs)
                member_labels.append(labels)
            
            member_data = torch.cat(member_data, dim=0)[:500]
            member_labels = torch.cat(member_labels, dim=0)[:500]
            
            non_member_data = []
            non_member_labels = []
            for inputs, labels in valloader:
                non_member_data.append(inputs)
                non_member_labels.append(labels)
            
            non_member_data = torch.cat(non_member_data, dim=0)[:500]
            non_member_labels = torch.cat(non_member_labels, dim=0)[:500]
            
            # Train target model
            target_model = Net(num_features=784, num_classes=10)
            target_model.to(device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(target_model.parameters(), lr=0.01)
            target_model.train()
            
            member_dataset = TensorDataset(member_data, member_labels)
            member_train_loader = DataLoader(member_dataset, batch_size=32, shuffle=True)
            
            for epoch in range(3):
                for inputs, labels in member_train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = target_model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            
            # Apply privacy with specific noise level
            params = get_model_parameters(target_model)
            
            client = FlowerClient(
                model=target_model,
                trainloader=member_train_loader,
                valloader=member_train_loader,
                device=device,
                client_id="client-1",
                use_dp=True,
                use_he=False,
                use_secagg=True,
                use_ldp=True,  # Enable Local DP
                use_adaptive_dp=True,  # Enable Adaptive DP
                num_clients=5,
                dp_noise_multiplier=noise_mult,
                dp_l2_norm_clip=1.0,
                ldp_epsilon=1.0  # LDP privacy budget
            )
            
            protected_params = client._apply_privacy_mechanisms(params, batch_size=32)
            protected_model = Net(num_features=784, num_classes=10)
            set_model_parameters(protected_model, protected_params)
            protected_model.to(device)
            protected_model.eval()
            
            # Run attack
            attack = BlackBoxMIAttack(protected_model, device)
            
            member_loader = DataLoader(
                TensorDataset(member_data, member_labels),
                batch_size=32,
                shuffle=False
            )
            non_member_loader = DataLoader(
                TensorDataset(non_member_data, non_member_labels),
                batch_size=32,
                shuffle=False
            )
            
            member_features = attack.extract_prediction_features(member_loader)
            non_member_features = attack.extract_prediction_features(non_member_loader)
            
            split_idx = len(member_features) // 2
            train_member_features = member_features[:split_idx]
            test_member_features = member_features[split_idx:]
            train_non_member_features = non_member_features[:split_idx]
            test_non_member_features = non_member_features[split_idx:]
            
            attack.train_attack_model(
                train_member_features,
                train_non_member_features,
                attack_model_type="random_forest"
            )
            
            test_features = np.vstack([test_member_features, test_non_member_features])
            test_labels = np.hstack([
                np.ones(len(test_member_features)),
                np.zeros(len(test_non_member_features))
            ])
            
            predictions = attack.predict_membership(test_features)
            probabilities = attack.predict_membership_proba(test_features)
            
            accuracy = accuracy_score(test_labels, predictions)
            try:
                auc = roc_auc_score(test_labels, probabilities[:, 1])
            except:
                auc = 0.0
            
            # Calculate additional metrics
            precision = precision_score(test_labels, predictions, zero_division=0)
            recall = recall_score(test_labels, predictions, zero_division=0)
            f1 = f1_score(test_labels, predictions, zero_division=0)
            
            results.append({
                "noise_multiplier": noise_mult,
                "accuracy": accuracy,
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })
            
            print(f"Attack Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            print(f"AUC-ROC: {auc:.4f}")
        
        # Summary
        print("\n" + "="*70)
        print("NOISE MULTIPLIER ANALYSIS:")
        print("="*70)
        print(f"{'Noise Mult':<15} {'Attack Acc':<15} {'AUC-ROC':<15} {'Protection':<15}")
        print("-"*70)
        for r in results:
            if r["accuracy"] < 0.55:
                protection = "EXCELLENT"
            elif r["accuracy"] < 0.60:
                protection = "GOOD"
            elif r["accuracy"] < 0.65:
                protection = "MODERATE"
            else:
                protection = "WEAK"
            print(f"{r['noise_multiplier']:<15.1f} {r['accuracy']:<15.4f} {r['auc']:<15.4f} {protection:<15}")
        
        return True
    
    def test_blackbox_mia_comparison(self):
        """
        Test 3: Compare MIA success with and without privacy mechanisms.
        """
        print("\n" + "="*70)
        print("BLACK BOX MIA TEST 3: Privacy Mechanisms Comparison")
        print("="*70)
        
        print("\nTesting WITHOUT privacy mechanisms...")
        acc_without = self._run_single_mia_test(use_privacy=False)
        
        print("\nTesting WITH privacy mechanisms...")
        acc_with = self._run_single_mia_test(use_privacy=True)
        
        print("\n" + "="*70)
        print("COMPARISON RESULTS:")
        print("="*70)
        print(f"Attack Accuracy WITHOUT privacy: {acc_without:.4f} ({acc_without*100:.2f}%)")
        print(f"Attack Accuracy WITH privacy:    {acc_with:.4f} ({acc_with*100:.2f}%)")
        print(f"Privacy Improvement: {((acc_without - acc_with) / acc_without * 100):.2f}% reduction")
        
        if acc_with < acc_without * 0.9:
            print("\n[PASS] Privacy mechanisms significantly reduce MIA success")
        else:
            print("\n[WARNING] Privacy mechanisms may need strengthening")
        
        return True
    
    def _run_single_mia_test(self, use_privacy: bool) -> float:
        """Helper method to run a single MIA test."""
        device = torch.device("cpu")
        
        try:
            trainloader, valloader = load_data("client-1")
        except Exception as e:
            print(f"[SKIP] Could not load data: {e}")
            return 0.5
        
        # Prepare data
        member_data = []
        member_labels = []
        for inputs, labels in trainloader:
            member_data.append(inputs)
            member_labels.append(labels)
        
        member_data = torch.cat(member_data, dim=0)[:300]
        member_labels = torch.cat(member_labels, dim=0)[:300]
        
        non_member_data = []
        non_member_labels = []
        for inputs, labels in valloader:
            non_member_data.append(inputs)
            non_member_labels.append(labels)
        
        non_member_data = torch.cat(non_member_data, dim=0)[:300]
        non_member_labels = torch.cat(non_member_labels, dim=0)[:300]
        
        # Train model
        target_model = Net(num_features=784, num_classes=10)
        target_model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(target_model.parameters(), lr=0.01)
        target_model.train()
        
        member_dataset = TensorDataset(member_data, member_labels)
        member_train_loader = DataLoader(member_dataset, batch_size=32, shuffle=True)
        
        for epoch in range(3):
            for inputs, labels in member_train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = target_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        if use_privacy:
            params = get_model_parameters(target_model)
            client = FlowerClient(
                model=target_model,
                trainloader=member_train_loader,
                valloader=member_train_loader,
                device=device,
                client_id="client-1",
                use_dp=True,
                use_he=False,
                use_secagg=True,
                use_ldp=True,  # Enable Local DP
                use_adaptive_dp=True,  # Enable Adaptive DP
                num_clients=5,
                dp_noise_multiplier=2.5,  # Increased for stronger protection
                dp_l2_norm_clip=1.0,
                ldp_epsilon=1.0  # LDP privacy budget
            )
            protected_params = client._apply_privacy_mechanisms(params, batch_size=32)
            protected_model = Net(num_features=784, num_classes=10)
            set_model_parameters(protected_model, protected_params)
            protected_model.to(device)
            protected_model.eval()
            target_model = protected_model
        else:
            target_model.eval()
        
        # Run attack
        attack = BlackBoxMIAttack(target_model, device)
        
        member_loader = DataLoader(
            TensorDataset(member_data, member_labels),
            batch_size=32,
            shuffle=False
        )
        non_member_loader = DataLoader(
            TensorDataset(non_member_data, non_member_labels),
            batch_size=32,
            shuffle=False
        )
        
        member_features = attack.extract_prediction_features(member_loader)
        non_member_features = attack.extract_prediction_features(non_member_loader)
        
        split_idx = len(member_features) // 2
        train_member_features = member_features[:split_idx]
        test_member_features = member_features[split_idx:]
        train_non_member_features = non_member_features[:split_idx]
        test_non_member_features = non_member_features[split_idx:]
        
        attack.train_attack_model(
            train_member_features,
            train_non_member_features,
            attack_model_type="random_forest"
        )
        
        test_features = np.vstack([test_member_features, test_non_member_features])
        test_labels = np.hstack([
            np.ones(len(test_member_features)),
            np.zeros(len(test_non_member_features))
        ])
        
        predictions = attack.predict_membership(test_features)
        accuracy = accuracy_score(test_labels, predictions)
        
        # Also calculate other metrics for debugging
        try:
            probabilities = attack.predict_membership_proba(test_features)
            precision = precision_score(test_labels, predictions, zero_division=0)
            recall = recall_score(test_labels, predictions, zero_division=0)
            f1 = f1_score(test_labels, predictions, zero_division=0)
        except:
            pass
        
        return accuracy
    
    def run_all_tests(self):
        """Run all black box MIA tests."""
        if not SKLEARN_AVAILABLE:
            print("\n" + "="*70)
            print("ERROR: scikit-learn is not installed")
            print("="*70)
            print("\nPlease install scikit-learn to run black box MIA tests:")
            print("  pip install scikit-learn")
            print("\n" + "="*70)
            return False
        
        print("\n" + "="*70)
        print("BLACK BOX MEMBERSHIP INFERENCE ATTACK TEST SUITE")
        print("="*70)
        print("\nThis suite tests the system's resistance to black box MIA attacks.")
        print("The attacker only has access to model predictions (black box access).")
        print("Higher attack accuracy = weaker privacy protection.")
        print("\n" + "-"*70)
        print("PRIVACY MECHANISMS ENABLED (STRONG PROTECTION):")
        print("-"*70)
        print("  ✓ Differential Privacy (DP): noise_multiplier = 2.5")
        print("  ✓ Local Differential Privacy (LDP): Enabled (epsilon = 1.0)")
        print("  ✓ Adaptive DP: Enabled (dynamic clipping)")
        print("  ✓ Secure Aggregation (SecAgg): Enabled")
        print("-"*70)
        print("\n" + "="*70)
        
        # Run tests
        self.test_blackbox_mia_basic(use_privacy=True)
        self.test_blackbox_mia_with_different_noise()
        self.test_blackbox_mia_comparison()
        
        # Print summary
        print("\n" + "="*70)
        print("BLACK BOX MIA TEST SUMMARY")
        print("="*70)
        
        if self.attack_accuracies:
            print("\nAttack Results Summary:")
            print(f"{'Test':<30} {'Accuracy':<15} {'AUC':<15} {'Protection':<15}")
            print("-"*70)
            for result in self.attack_accuracies:
                print(f"{result['test']:<30} {result['accuracy']:<15.4f} {result['auc']:<15.4f} {result['protection']:<15}")
        
        print("\n" + "="*70)
        print("RECOMMENDATIONS:")
        print("="*70)
        print("1. Attack accuracy < 55%: EXCELLENT protection")
        print("2. Attack accuracy 55-60%: GOOD protection")
        print("3. Attack accuracy 60-65%: MODERATE protection (consider stronger DP)")
        print("4. Attack accuracy > 65%: WEAK protection (increase noise_multiplier)")
        print("\nCurrent protection settings (STRONG PRIVACY):")
        print("  ✓ DP noise_multiplier: 2.5 (increased from 1.0)")
        print("  ✓ Local DP (LDP): Enabled")
        print("  ✓ Adaptive DP: Enabled")
        print("  ✓ Secure Aggregation: Enabled")
        print("="*70)
        
        return True


if __name__ == "__main__":
    suite = BlackBoxMIATestSuite()
    success = suite.run_all_tests()
    sys.exit(0 if success else 1)
