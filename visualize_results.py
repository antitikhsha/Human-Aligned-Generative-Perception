"""
Visualization of VGG vs HPE Results

Generate comparison plots showing improvement from VGG baseline to HPE
"""
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


def load_results(results_dir):
    """Load VGG and HPE results"""
    vgg_path = os.path.join(results_dir, 'vgg_baseline_results.json')
    hpe_path = os.path.join(results_dir, 'hpe_results.json')
    
    vgg_results = None
    hpe_results = None
    
    if os.path.exists(vgg_path):
        with open(vgg_path, 'r') as f:
            vgg_results = json.load(f)
    
    if os.path.exists(hpe_path):
        with open(hpe_path, 'r') as f:
            hpe_results = json.load(f)
    
    return vgg_results, hpe_results


def plot_kendall_tau_comparison(vgg_results, hpe_results, output_dir):
    """Create bar plot comparing Kendall Tau"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = []
    taus = []
    colors = []
    
    if vgg_results:
        methods.append('VGG19\nPerceptual Loss')
        taus.append(vgg_results['kendall_tau'])
        colors.append('#E74C3C')  # Red
    
    if hpe_results:
        methods.append('HPE\n(ResNet18 + Triplet Loss)')
        taus.append(hpe_results['kendall_tau'])
        colors.append('#27AE60')  # Green
    
    bars = ax.bar(methods, taus, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, tau in zip(bars, taus):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{tau:.4f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add target line
    ax.axhline(y=0.82, color='blue', linestyle='--', linewidth=2, label='Target (0.82)')
    
    # Calculate improvement if both results available
    if vgg_results and hpe_results:
        improvement = ((hpe_results['kendall_tau'] - vgg_results['kendall_tau']) / 
                      vgg_results['kendall_tau']) * 100
        ax.text(0.5, 0.95, f'Improvement: {improvement:.1f}%',
                transform=ax.transAxes, ha='center', fontsize=16,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_ylabel('Kendall Tau Correlation', fontsize=14)
    ax.set_title('Human Perception Alignment: VGG vs HPE', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kendall_tau_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: kendall_tau_comparison.png")
    plt.close()


def plot_training_history(results_dir, output_dir):
    """Plot training loss curves"""
    history_path = os.path.join(results_dir, 'training_history.json')
    
    if not os.path.exists(history_path):
        print("Training history not found, skipping loss curves")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    train_losses = history['train_losses']
    val_losses = history['val_losses']
    epochs = range(1, len(train_losses) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, train_losses, 'o-', label='Training Loss', color='#3498DB', linewidth=2)
    ax.plot(epochs, val_losses, 's-', label='Validation Loss', color='#E74C3C', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Triplet Loss', fontsize=14)
    ax.set_title('HPE Training Progress', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: training_curves.png")
    plt.close()


def plot_accuracy_comparison(vgg_results, hpe_results, output_dir):
    """Create bar plot comparing accuracy"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = []
    accuracies = []
    colors = []
    
    if vgg_results:
        methods.append('VGG19')
        accuracies.append(vgg_results['accuracy'] * 100)
        colors.append('#E74C3C')
    
    if hpe_results:
        methods.append('HPE')
        accuracies.append(hpe_results['accuracy'] * 100)
        colors.append('#27AE60')
    
    bars = ax.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('Triplet Prediction Accuracy', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: accuracy_comparison.png")
    plt.close()


def create_summary_report(vgg_results, hpe_results, output_dir):
    """Create text summary report"""
    report = []
    report.append("="*60)
    report.append("HUMAN-ALIGNED PERCEPTUAL EMBEDDING - RESULTS SUMMARY")
    report.append("="*60)
    report.append("")
    
    if vgg_results:
        report.append("Phase 1: VGG Baseline Evaluation")
        report.append("-" * 40)
        report.append(f"  Method: {vgg_results['method']}")
        report.append(f"  Triplets evaluated: {vgg_results['num_triplets']:,}")
        report.append(f"  Kendall Tau: {vgg_results['kendall_tau']:.4f}")
        report.append(f"  Accuracy: {vgg_results['accuracy']*100:.2f}%")
        report.append("")
    
    if hpe_results:
        report.append("Phase 2: HPE Model Evaluation")
        report.append("-" * 40)
        report.append(f"  Method: {hpe_results['method']}")
        report.append(f"  Triplets evaluated: {hpe_results['num_triplets']:,}")
        report.append(f"  Kendall Tau: {hpe_results['kendall_tau']:.4f}")
        report.append(f"  Accuracy: {hpe_results['accuracy']*100:.2f}%")
        report.append(f"  Embedding dim: {hpe_results['embedding_dim']}")
        report.append("")
    
    if vgg_results and hpe_results:
        improvement = ((hpe_results['kendall_tau'] - vgg_results['kendall_tau']) / 
                      vgg_results['kendall_tau']) * 100
        report.append("Improvement")
        report.append("-" * 40)
        report.append(f"  VGG Tau: {vgg_results['kendall_tau']:.4f}")
        report.append(f"  HPE Tau: {hpe_results['kendall_tau']:.4f}")
        report.append(f"  Improvement: {improvement:.1f}%")
        report.append(f"  Target improvement: ≥30%")
        report.append("")
        
        if improvement >= 30:
            report.append("  ✓ SUCCESS: Achieved target improvement!")
        elif improvement >= 25:
            report.append("  ✓ Good: Close to target")
        else:
            report.append("  ⚠ Below target improvement")
        report.append("")
    
    report.append("="*60)
    
    report_text = "\n".join(report)
    print(report_text)
    
    # Save to file
    with open(os.path.join(output_dir, 'results_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\nSaved: results_summary.txt")


def main():
    parser = argparse.ArgumentParser(description='Visualize VGG vs HPE results')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--output_dir', type=str, default='results')
    
    args = parser.parse_args()
    
    print("Loading results...")
    vgg_results, hpe_results = load_results(args.results_dir)
    
    if not vgg_results and not hpe_results:
        print("No results found! Please run evaluation scripts first.")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    # Create plots
    if vgg_results and hpe_results:
        plot_kendall_tau_comparison(vgg_results, hpe_results, args.output_dir)
        plot_accuracy_comparison(vgg_results, hpe_results, args.output_dir)
    
    plot_training_history(args.results_dir, args.output_dir)
    
    # Create summary
    create_summary_report(vgg_results, hpe_results, args.output_dir)
    
    print("\n✓ All visualizations complete!")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
