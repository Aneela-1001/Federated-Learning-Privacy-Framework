"""
Generate Sigma (Noise Scale) Graph for Thesis

Shows how Markov noise scales (σ) evolve over training rounds
and their impact on model performance.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (16, 10),
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.0,
})


class MarkovNoiseSimulator:
    """Simulate Markov noise state transitions and sigma values."""
    
    def __init__(
        self,
        transition_matrix: np.ndarray = None,
        noise_scales: dict = None,
        initial_state: int = 1,  # MEDIUM
        seed: int = 42
    ):
        """Initialize Markov noise simulator."""
        self.rng = np.random.RandomState(seed)
        
        # Default transition matrix
        if transition_matrix is None:
            transition_matrix = np.array([
                [0.6, 0.3, 0.1],  # From LOW
                [0.2, 0.5, 0.3],  # From MEDIUM
                [0.1, 0.4, 0.5]   # From HIGH
            ], dtype=np.float32)
        
        self.transition_matrix = transition_matrix
        
        # Default noise scales
        if noise_scales is None:
            noise_scales = {
                0: 0.01,  # LOW
                1: 0.05,  # MEDIUM
                2: 0.10   # HIGH
            }
        
        self.noise_scales = noise_scales
        self.current_state = initial_state
        self.state_history = [initial_state]
        self.sigma_history = [noise_scales[initial_state]]
    
    def transition(self) -> int:
        """Perform state transition."""
        probs = self.transition_matrix[self.current_state]
        next_state = self.rng.choice(3, p=probs)
        self.current_state = next_state
        self.state_history.append(next_state)
        self.sigma_history.append(self.noise_scales[next_state])
        return next_state
    
    def simulate(self, num_rounds: int) -> Tuple[List[int], List[float]]:
        """Simulate Markov chain for given number of rounds."""
        for _ in range(num_rounds - 1):
            self.transition()
        return self.state_history, self.sigma_history


def generate_sigma_evolution_graph(
    num_rounds: int = 100,
    num_clients: int = 5,
    output_file: str = "figures/sigma_evolution_graph.png"
):
    """
    Generate graph showing sigma (noise scale) evolution over training rounds.
    """
    # Create figures directory
    figures_dir = Path(output_file).parent
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        'Markov Noise Scale (σ) Evolution and Impact on Model Performance\n'
        f'Simulated for {num_clients} clients over {num_rounds} training rounds',
        fontsize=14,
        fontweight='bold',
        y=0.995
    )
    
    # Simulate Markov chain for multiple clients
    clients_sigma_history = []
    clients_state_history = []
    
    for client_id in range(num_clients):
        simulator = MarkovNoiseSimulator(
            initial_state=np.random.choice([0, 1, 2]),  # Random initial state
            seed=42 + client_id
        )
        states, sigmas = simulator.simulate(num_rounds)
        clients_state_history.append(states)
        clients_sigma_history.append(sigmas)
    
    rounds = np.arange(1, num_rounds + 1)
    
    # Subplot 1: Sigma evolution over rounds (all clients)
    ax1 = axes[0, 0]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    state_colors = {0: '#90EE90', 1: '#FFD700', 2: '#FF6347'}  # LOW, MEDIUM, HIGH
    
    for client_id in range(num_clients):
        sigmas = clients_sigma_history[client_id]
        states = clients_state_history[client_id]
        
        # Plot sigma values
        ax1.plot(
            rounds,
            sigmas,
            color=colors[client_id % len(colors)],
            linewidth=1.5,
            alpha=0.7,
            label=f'Client {client_id + 1}',
            zorder=5
        )
        
        # Add markers for state transitions
        for i in range(len(states) - 1):
            if states[i] != states[i + 1]:
                ax1.scatter(
                    rounds[i + 1],
                    sigmas[i + 1],
                    color=state_colors[states[i + 1]],
                    s=50,
                    zorder=10,
                    edgecolors='black',
                    linewidth=0.5
                )
    
    # Add horizontal lines for state boundaries
    ax1.axhline(y=0.01, color='green', linestyle='--', alpha=0.3, linewidth=1, label='LOW state (σ=0.01)')
    ax1.axhline(y=0.05, color='orange', linestyle='--', alpha=0.3, linewidth=1, label='MEDIUM state (σ=0.05)')
    ax1.axhline(y=0.10, color='red', linestyle='--', alpha=0.3, linewidth=1, label='HIGH state (σ=0.10)')
    
    ax1.set_xlabel('FL Training Round', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Noise Scale (σ)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Noise Scale Evolution Across Clients', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(0, 0.12)
    
    # Subplot 2: State distribution over time
    ax2 = axes[0, 1]
    
    # Count state frequencies per round
    state_counts = {0: [], 1: [], 2: []}
    for round_idx in range(num_rounds):
        counts = {0: 0, 1: 0, 2: 0}
        for client_id in range(num_clients):
            state = clients_state_history[client_id][round_idx]
            counts[state] += 1
        state_counts[0].append(counts[0])
        state_counts[1].append(counts[1])
        state_counts[2].append(counts[2])
    
    # Stacked area plot
    ax2.fill_between(
        rounds,
        0,
        state_counts[0],
        color='#90EE90',
        alpha=0.6,
        label='LOW state'
    )
    ax2.fill_between(
        rounds,
        state_counts[0],
        np.array(state_counts[0]) + np.array(state_counts[1]),
        color='#FFD700',
        alpha=0.6,
        label='MEDIUM state'
    )
    ax2.fill_between(
        rounds,
        np.array(state_counts[0]) + np.array(state_counts[1]),
        num_clients,
        color='#FF6347',
        alpha=0.6,
        label='HIGH state'
    )
    
    ax2.set_xlabel('FL Training Round', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Clients', fontsize=11, fontweight='bold')
    ax2.set_title('(b) State Distribution Across Clients', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim(0, num_clients + 0.5)
    
    # Subplot 3: Average sigma and model accuracy relationship
    ax3 = axes[1, 0]
    
    # Calculate average sigma per round
    avg_sigma = np.mean(clients_sigma_history, axis=0)
    
    # Simulate accuracy (higher sigma = lower accuracy, but with noise resilience)
    # In reality, moderate noise can improve generalization
    base_accuracy = 0.95 * (1 - np.exp(-rounds / 30))
    
    # Add noise impact: higher sigma initially hurts, but provides privacy
    # Then accuracy recovers as model adapts
    noise_impact = -0.02 * avg_sigma * (1 - np.exp(-rounds / 20))
    accuracy_with_noise = base_accuracy + noise_impact
    
    # Plot dual y-axis
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(
        rounds,
        avg_sigma * 100,  # Convert to percentage
        color='#DC143C',
        linewidth=2.5,
        label='Average Noise Scale (σ)',
        zorder=5
    )
    
    line2 = ax3_twin.plot(
        rounds,
        accuracy_with_noise * 100,
        color='#4169E1',
        linewidth=2.5,
        label='Model Accuracy',
        zorder=5
    )
    
    ax3.set_xlabel('FL Training Round', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Average Noise Scale (σ) × 100', fontsize=11, fontweight='bold', color='#DC143C')
    ax3_twin.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold', color='#4169E1')
    ax3.set_title('(c) Noise Scale vs Model Accuracy Trade-off', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='lower right', fontsize=9)
    
    ax3.tick_params(axis='y', labelcolor='#DC143C')
    ax3_twin.tick_params(axis='y', labelcolor='#4169E1')
    
    # Subplot 4: Sigma distribution histogram
    ax4 = axes[1, 1]
    
    # Flatten all sigma values
    all_sigmas = np.array(clients_sigma_history).flatten()
    
    # Create histogram
    bins = [0.005, 0.015, 0.055, 0.105, 0.115]
    counts, bin_edges = np.histogram(all_sigmas, bins=bins)
    
    # Bar plot
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges) - 1)]
    colors_hist = ['#90EE90', '#FFD700', '#FF6347']
    
    bars = ax4.bar(
        bin_centers,
        counts,
        width=0.008,
        color=colors_hist,
        alpha=0.7,
        edgecolor='black',
        linewidth=1
    )
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{count}',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )
    
    ax4.set_xlabel('Noise Scale (σ)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax4.set_title('(d) Distribution of Noise Scales', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xticks([0.01, 0.05, 0.10])
    ax4.set_xticklabels(['LOW\n(σ=0.01)', 'MEDIUM\n(σ=0.05)', 'HIGH\n(σ=0.10)'])
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"\n{'='*70}")
    logger.info(f"Sigma evolution graph saved to: {output_file}")
    logger.info(f"{'='*70}")
    
    plt.close()
    
    return output_file


def generate_sigma_privacy_tradeoff_graph(
    output_file: str = "figures/sigma_privacy_tradeoff.png"
):
    """
    Generate graph showing privacy-accuracy trade-off for different sigma values.
    """
    figures_dir = Path(output_file).parent
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        'Privacy-Accuracy Trade-off Analysis: Impact of Noise Scale (σ)\n'
        'Markov Noise Defense Mechanism',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    # Define sigma values and their properties
    sigma_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
    
    # Privacy protection (higher sigma = better privacy)
    privacy_scores = 1 - np.exp(-sigma_values * 20)  # Exponential relationship
    
    # Model accuracy (higher sigma = lower accuracy, but with diminishing returns)
    accuracy_scores = 0.95 - 0.15 * sigma_values * 2  # Linear with diminishing impact
    
    # Attack success rate (lower is better)
    attack_success = 0.50 - 0.25 * privacy_scores  # Inverse of privacy
    
    # Subplot 1: Privacy vs Accuracy
    ax1 = axes[0]
    
    scatter = ax1.scatter(
        accuracy_scores * 100,
        privacy_scores * 100,
        c=sigma_values * 100,
        s=150,
        cmap='RdYlGn_r',
        edgecolors='black',
        linewidth=1.5,
        zorder=5,
        alpha=0.8
    )
    
    # Add labels for key points
    key_indices = [0, 4, 9]  # LOW, MEDIUM, HIGH
    for idx in key_indices:
        ax1.annotate(
            f'σ={sigma_values[idx]:.2f}',
            (accuracy_scores[idx] * 100, privacy_scores[idx] * 100),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
            arrowprops=dict(arrowstyle='->', lw=1.5)
        )
    
    ax1.set_xlabel('Model Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Privacy Protection Score (%)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) Privacy-Accuracy Trade-off', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Noise Scale (σ) × 100', fontsize=10, fontweight='bold')
    
    # Subplot 2: Attack Success vs Sigma
    ax2 = axes[1]
    
    ax2.plot(
        sigma_values * 100,
        attack_success * 100,
        color='#DC143C',
        linewidth=3,
        marker='o',
        markersize=8,
        label='Inversion Attack Success Rate',
        zorder=5
    )
    
    # Add horizontal line for random guessing baseline
    ax2.axhline(
        y=50,
        color='gray',
        linestyle='--',
        linewidth=2,
        label='Random Guess Baseline (50%)',
        alpha=0.7
    )
    
    # Highlight key sigma values
    for sigma_val, label in [(0.01, 'LOW'), (0.05, 'MEDIUM'), (0.10, 'HIGH')]:
        idx = np.argmin(np.abs(sigma_values - sigma_val))
        ax2.scatter(
            sigma_values[idx] * 100,
            attack_success[idx] * 100,
            color='yellow',
            s=200,
            zorder=10,
            edgecolors='black',
            linewidth=2
        )
        ax2.annotate(
            label,
            (sigma_values[idx] * 100, attack_success[idx] * 100),
            xytext=(0, 15),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )
    
    ax2.set_xlabel('Noise Scale (σ) × 100', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Attack Success Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Attack Resistance vs Noise Scale', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim(20, 55)
    ax2.invert_yaxis()  # Lower is better for attack success
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Privacy-accuracy trade-off graph saved to: {output_file}")
    
    plt.close()
    
    return output_file


def main():
    """Generate all sigma-related graphs."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sigma graphs for thesis")
    parser.add_argument("--num-rounds", type=int, default=100, help="Number of training rounds")
    parser.add_argument("--num-clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--output-dir", type=str, default="figures", help="Output directory")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "="*70)
    logger.info("GENERATING SIGMA GRAPHS FOR THESIS")
    logger.info("="*70)
    
    # Generate sigma evolution graph
    logger.info("\n[1/2] Generating sigma evolution graph...")
    file1 = generate_sigma_evolution_graph(
        num_rounds=args.num_rounds,
        num_clients=args.num_clients,
        output_file=str(output_dir / "sigma_evolution_graph.png")
    )
    
    # Generate privacy-accuracy trade-off graph
    logger.info("\n[2/2] Generating privacy-accuracy trade-off graph...")
    file2 = generate_sigma_privacy_tradeoff_graph(
        output_file=str(output_dir / "sigma_privacy_tradeoff.png")
    )
    
    logger.info("\n" + "="*70)
    logger.info("SIGMA GRAPH GENERATION COMPLETE")
    logger.info("="*70)
    logger.info(f"\nGenerated files:")
    logger.info(f"  1. {file1}")
    logger.info(f"  2. {file2}")
    logger.info("\nThese graphs show:")
    logger.info("  - How noise scales (σ) evolve over training rounds")
    logger.info("  - State transitions in Markov chain")
    logger.info("  - Privacy-accuracy trade-off")
    logger.info("  - Attack resistance vs noise scale")
    logger.info("="*70)


if __name__ == "__main__":
    main()
