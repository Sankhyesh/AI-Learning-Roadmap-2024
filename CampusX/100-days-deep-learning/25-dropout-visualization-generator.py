"""
Dropout Visualization Generator
This script creates visualizations for understanding dropout layers in neural networks
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.patches as mpatches

def create_dropout_concept_visualization():
    """Creates a visualization showing how dropout works"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Network architecture
    layers = [3, 6, 4, 2]  # neurons per layer
    
    # Plot 1: Without Dropout
    ax1.set_title("Without Dropout", fontsize=14, fontweight='bold')
    ax1.set_xlim(-0.5, 3.5)
    ax1.set_ylim(-0.5, 6.5)
    
    # Draw neurons and connections
    neuron_positions = {}
    for layer_idx, n_neurons in enumerate(layers):
        y_positions = np.linspace(1, 5, n_neurons)
        for neuron_idx, y_pos in enumerate(y_positions):
            # Store position
            neuron_positions[(layer_idx, neuron_idx)] = (layer_idx, y_pos)
            # Draw neuron
            circle = Circle((layer_idx, y_pos), 0.15, color='blue', ec='black', lw=2)
            ax1.add_patch(circle)
            
            # Draw connections to next layer
            if layer_idx < len(layers) - 1:
                next_layer_positions = np.linspace(1, 5, layers[layer_idx + 1])
                for next_y in next_layer_positions:
                    ax1.plot([layer_idx, layer_idx + 1], [y_pos, next_y], 
                            'k-', alpha=0.3, lw=1)
    
    # Plot 2: With Dropout (p=0.5)
    ax2.set_title("With Dropout (p=0.5)", fontsize=14, fontweight='bold')
    ax2.set_xlim(-0.5, 3.5)
    ax2.set_ylim(-0.5, 6.5)
    
    # Randomly drop neurons
    np.random.seed(42)
    dropout_rate = 0.5
    
    for layer_idx, n_neurons in enumerate(layers):
        y_positions = np.linspace(1, 5, n_neurons)
        for neuron_idx, y_pos in enumerate(y_positions):
            # Don't drop neurons in input or output layers
            if layer_idx == 0 or layer_idx == len(layers) - 1:
                is_dropped = False
            else:
                is_dropped = np.random.random() < dropout_rate
            
            # Draw neuron
            color = 'lightgray' if is_dropped else 'blue'
            alpha = 0.3 if is_dropped else 1.0
            circle = Circle((layer_idx, y_pos), 0.15, color=color, 
                          ec='black', lw=2, alpha=alpha)
            ax2.add_patch(circle)
            
            # Draw connections from this neuron if not dropped
            if layer_idx < len(layers) - 1 and not is_dropped:
                next_layer_positions = np.linspace(1, 5, layers[layer_idx + 1])
                for next_idx, next_y in enumerate(next_layer_positions):
                    # Check if next neuron is dropped
                    if layer_idx + 1 < len(layers) - 1:
                        next_is_dropped = np.random.random() < dropout_rate
                    else:
                        next_is_dropped = False
                    
                    if not next_is_dropped or layer_idx + 1 == len(layers) - 1:
                        ax2.plot([layer_idx, layer_idx + 1], [y_pos, next_y], 
                                'k-', alpha=0.3 if not is_dropped else 0.1, lw=1)
    
    # Add labels
    for ax in [ax1, ax2]:
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(['Input', 'Hidden 1', 'Hidden 2', 'Output'])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    # Add legend
    active_patch = mpatches.Patch(color='blue', label='Active neuron')
    dropped_patch = mpatches.Patch(color='lightgray', label='Dropped neuron')
    ax2.legend(handles=[active_patch, dropped_patch], loc='upper right')
    
    plt.tight_layout()
    return fig

def create_overfitting_visualization():
    """Creates visualization showing overfitting in regression and classification"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Regression overfitting
    np.random.seed(42)
    X_train = np.linspace(0, 1, 20)
    y_train = np.sin(2 * np.pi * X_train) + np.random.normal(0, 0.1, 20)
    X_test = np.linspace(0, 1, 10) + 0.05
    y_test = np.sin(2 * np.pi * X_test) + np.random.normal(0, 0.1, 10)
    
    # Overfitted model (high degree polynomial)
    X_plot = np.linspace(0, 1, 200)
    coeffs = np.polyfit(X_train, y_train, 15)
    y_overfit = np.polyval(coeffs, X_plot)
    
    ax1.scatter(X_train, y_train, c='black', s=50, label='Training data', zorder=5)
    ax1.scatter(X_test, y_test, c='red', s=50, label='Test data', alpha=0.7, zorder=5)
    ax1.plot(X_plot, y_overfit, 'b-', linewidth=2, label='Overfitted curve')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Overfitting in Regression', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Classification overfitting
    np.random.seed(42)
    n_points = 100
    X_class = np.random.randn(n_points, 2)
    y_class = (X_class[:, 0] + X_class[:, 1] > 0).astype(int)
    # Add some noise
    flip_indices = np.random.choice(n_points, 10, replace=False)
    y_class[flip_indices] = 1 - y_class[flip_indices]
    
    ax2.scatter(X_class[y_class == 0, 0], X_class[y_class == 0, 1], 
               c='blue', s=50, alpha=0.7, label='Class 0')
    ax2.scatter(X_class[y_class == 1, 0], X_class[y_class == 1, 1], 
               c='orange', s=50, alpha=0.7, label='Class 1')
    
    # Draw irregular decision boundary
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    # Create irregular boundary
    Z = np.zeros_like(xx)
    for i in range(len(X_class)):
        if y_class[i] == 1:
            dist = np.sqrt((xx - X_class[i, 0])**2 + (yy - X_class[i, 1])**2)
            Z += np.exp(-5 * dist)
    
    ax2.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_title('Overfitting in Classification', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_dropout_rate_comparison():
    """Creates comparison of different dropout rates"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Generate classification data
    np.random.seed(42)
    n_points = 100
    X = np.random.randn(n_points, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    # Add noise
    flip_indices = np.random.choice(n_points, 10, replace=False)
    y[flip_indices] = 1 - y[flip_indices]
    
    dropout_rates = [0.0, 0.2, 0.5]
    titles = ['No Dropout (p=0.0)', 'Light Dropout (p=0.2)', 'Moderate Dropout (p=0.5)']
    
    for ax, p, title in zip(axes, dropout_rates, titles):
        # Plot data points
        ax.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', s=50, alpha=0.7)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c='orange', s=50, alpha=0.7)
        
        # Create decision boundary based on dropout rate
        xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
        
        if p == 0.0:
            # Very irregular boundary
            Z = np.zeros_like(xx)
            for i in range(len(X)):
                if y[i] == 1:
                    dist = np.sqrt((xx - X[i, 0])**2 + (yy - X[i, 1])**2)
                    Z += np.exp(-5 * dist)
        elif p == 0.2:
            # Somewhat smooth boundary
            Z = xx + yy + 0.3 * np.sin(5 * xx) * np.cos(5 * yy)
        else:
            # Very smooth boundary
            Z = xx + yy
        
        ax.contour(xx, yy, Z, levels=[0.5 if p > 0 else np.median(Z)], 
                  colors='black', linewidths=2)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
    
    plt.tight_layout()
    return fig

def create_loss_curves():
    """Creates loss curves comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = np.arange(1, 101)
    
    # Without dropout
    train_loss_no_dropout = 0.5 * np.exp(-0.05 * epochs) + 0.01
    val_loss_no_dropout = 0.5 * np.exp(-0.02 * epochs) + 0.04 + 0.001 * epochs
    
    ax1.plot(epochs, train_loss_no_dropout, 'b-', linewidth=2, label='Training Loss')
    ax1.plot(epochs, val_loss_no_dropout, 'r-', linewidth=2, label='Validation Loss')
    ax1.fill_between(epochs, train_loss_no_dropout, val_loss_no_dropout, 
                     alpha=0.3, color='gray', label='Overfitting Gap')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Without Dropout', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 0.6)
    
    # With dropout
    train_loss_dropout = 0.5 * np.exp(-0.03 * epochs) + 0.02
    val_loss_dropout = 0.5 * np.exp(-0.025 * epochs) + 0.025
    
    ax2.plot(epochs, train_loss_dropout, 'b-', linewidth=2, label='Training Loss')
    ax2.plot(epochs, val_loss_dropout, 'r-', linewidth=2, label='Validation Loss')
    ax2.fill_between(epochs, train_loss_dropout, val_loss_dropout, 
                     alpha=0.3, color='lightgreen', label='Small Gap')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('With Dropout (p=0.3)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 0.6)
    
    plt.tight_layout()
    return fig

def save_all_visualizations():
    """Generate and save all visualizations"""
    import os
    
    # Create directory if it doesn't exist
    save_dir = "images/25"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Generate and save each visualization
    print("Generating dropout concept visualization...")
    fig1 = create_dropout_concept_visualization()
    fig1.savefig(f"{save_dir}/dropout-concept.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    print("Generating overfitting visualization...")
    fig2 = create_overfitting_visualization()
    fig2.savefig(f"{save_dir}/overfitting-examples.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    print("Generating dropout rate comparison...")
    fig3 = create_dropout_rate_comparison()
    fig3.savefig(f"{save_dir}/dropout-rate-comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    print("Generating loss curves comparison...")
    fig4 = create_loss_curves()
    fig4.savefig(f"{save_dir}/loss-curves-comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig4)
    
    print("All visualizations saved successfully!")

if __name__ == "__main__":
    # Uncomment to generate actual image files
    # save_all_visualizations()
    
    # For demonstration, just show the plots
    import matplotlib.pyplot as plt
    
    fig1 = create_dropout_concept_visualization()
    fig2 = create_overfitting_visualization()
    fig3 = create_dropout_rate_comparison()
    fig4 = create_loss_curves()
    
    plt.show()