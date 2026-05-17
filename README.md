# DDFed-Markov

A robust privacy-preserving federated learning framework integrating Markovian noise injection, Fully Homomorphic Encryption (FHE), and consensus-based validation to defend against privacy leakage and poisoning attacks.

## Overview

DDFed-Markov is a dual-defense federated learning framework designed to improve both privacy and robustness in decentralized machine learning environments.

The framework combines:

- Markovian probabilistic noise injection
- Fully Homomorphic Encryption (FHE)
- Secure aggregation
- Consensus-based malicious client detection

Unlike traditional federated learning systems, DDFed-Markov protects against:

- Model inversion attacks
- Data reconstruction attacks
- Poisoning attacks
- Byzantine attacks

while maintaining strong model performance.

---

## Key Features

- Privacy-preserving federated learning
- Markov-chain based adaptive noise injection
- CKKS homomorphic encryption
- Consensus-driven anomaly detection
- Defense against IPM, ALIE, and Scaling attacks
- Compatible with decentralized FL environments

---

## System Architecture

The framework consists of three major stages:

1. Client-side local training
2. Secure encrypted aggregation
3. Consensus-based validation

### Core Components

- Markovian Noise Injection
- FHE-based Secure Aggregation
- Client-side Validation
- Adaptive Consensus Filtering

---

## Technologies Used

- Python
- PyTorch
- Flower Framework
- NumPy
- CKKS Homomorphic Encryption
- Federated Learning

---

## Dataset

Experiments were performed on:

- MNIST
- Fashion-MNIST (FMNIST)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/DDFed-Markov.git
cd DDFed-Markov
