"""
Regularization in Deep Learning - Practical Implementation
This code demonstrates L1, L2, and Elastic Net regularization in neural networks
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

def create_dataset():
    """Create a non-linear classification dataset"""
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def create_model_without_regularization():
    """Create a model prone to overfitting"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(2,)),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def create_model_with_l2_regularization(lambda_value=0.01):
    """Create a model with L2 regularization"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(2,), 
              kernel_regularizer=regularizers.l2(lambda_value)),
        Dense(128, activation='relu', 
              kernel_regularizer=regularizers.l2(lambda_value)),
        Dense(1, activation='sigmoid')  # No regularization on output layer
    ])
    return model

def create_model_with_l1_regularization(lambda_value=0.01):
    """Create a model with L1 regularization"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(2,), 
              kernel_regularizer=regularizers.l1(lambda_value)),
        Dense(128, activation='relu', 
              kernel_regularizer=regularizers.l1(lambda_value)),
        Dense(1, activation='sigmoid')
    ])
    return model

def plot_decision_boundary(model, X, y, title):
    """Plot the decision boundary of a trained model"""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.show()

def plot_weight_distribution(weights1, weights2, labels):
    """Compare weight distributions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Box plots
    ax1.boxplot([weights1.flatten(), weights2.flatten()], labels=labels)
    ax1.set_title('Weight Distribution Comparison')
    ax1.set_ylabel('Weight Values')
    ax1.grid(True, alpha=0.3)
    
    # Probability density
    ax2.hist(weights1.flatten(), bins=50, alpha=0.5, label=labels[0], density=True)
    ax2.hist(weights2.flatten(), bins=50, alpha=0.5, label=labels[1], density=True)
    ax2.set_xlabel('Weight Values')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Weight Distribution (PDF)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_regularization_effects():
    """Main function to demonstrate regularization effects"""
    
    # Create dataset
    X_train, X_test, y_train, y_test = create_dataset()
    
    # 1. Train model without regularization
    print("Training model without regularization...")
    model_no_reg = create_model_without_regularization()
    model_no_reg.compile(optimizer=Adam(0.01), loss='binary_crossentropy', metrics=['accuracy'])
    history_no_reg = model_no_reg.fit(X_train, y_train, 
                                      validation_data=(X_test, y_test),
                                      epochs=200, batch_size=32, verbose=0)
    
    # 2. Train model with L2 regularization
    print("Training model with L2 regularization (λ=0.01)...")
    model_l2 = create_model_with_l2_regularization(0.01)
    model_l2.compile(optimizer=Adam(0.01), loss='binary_crossentropy', metrics=['accuracy'])
    history_l2 = model_l2.fit(X_train, y_train, 
                             validation_data=(X_test, y_test),
                             epochs=200, batch_size=32, verbose=0)
    
    # 3. Train model with L1 regularization
    print("Training model with L1 regularization (λ=0.01)...")
    model_l1 = create_model_with_l1_regularization(0.01)
    model_l1.compile(optimizer=Adam(0.01), loss='binary_crossentropy', metrics=['accuracy'])
    history_l1 = model_l1.fit(X_train, y_train, 
                             validation_data=(X_test, y_test),
                             epochs=200, batch_size=32, verbose=0)
    
    # Plot decision boundaries
    plot_decision_boundary(model_no_reg, X_test, y_test, 
                          'Decision Boundary - No Regularization')
    plot_decision_boundary(model_l2, X_test, y_test, 
                          'Decision Boundary - L2 Regularization (λ=0.01)')
    plot_decision_boundary(model_l1, X_test, y_test, 
                          'Decision Boundary - L1 Regularization (λ=0.01)')
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history_no_reg.history['loss'], label='Train Loss')
    plt.plot(history_no_reg.history['val_loss'], label='Val Loss')
    plt.title('No Regularization')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history_l2.history['loss'], label='Train Loss')
    plt.plot(history_l2.history['val_loss'], label='Val Loss')
    plt.title('L2 Regularization')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(history_l1.history['loss'], label='Train Loss')
    plt.plot(history_l1.history['val_loss'], label='Val Loss')
    plt.title('L1 Regularization')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Compare weight distributions
    weights_no_reg = model_no_reg.layers[0].get_weights()[0]
    weights_l2 = model_l2.layers[0].get_weights()[0]
    weights_l1 = model_l1.layers[0].get_weights()[0]
    
    plot_weight_distribution(weights_no_reg, weights_l2, 
                           ['No Regularization', 'L2 Regularization'])
    plot_weight_distribution(weights_no_reg, weights_l1, 
                           ['No Regularization', 'L1 Regularization'])
    
    # Print weight statistics
    print("\nWeight Statistics:")
    print(f"No Regularization - Min: {weights_no_reg.min():.3f}, Max: {weights_no_reg.max():.3f}")
    print(f"L2 Regularization - Min: {weights_l2.min():.3f}, Max: {weights_l2.max():.3f}")
    print(f"L1 Regularization - Min: {weights_l1.min():.3f}, Max: {weights_l1.max():.3f}")
    
    # Count near-zero weights for L1 (sparsity check)
    threshold = 0.01
    l1_sparse = np.sum(np.abs(weights_l1) < threshold)
    no_reg_sparse = np.sum(np.abs(weights_no_reg) < threshold)
    print(f"\nNear-zero weights (|w| < {threshold}):")
    print(f"No Regularization: {no_reg_sparse}/{weights_no_reg.size}")
    print(f"L1 Regularization: {l1_sparse}/{weights_l1.size}")

def experiment_with_lambda_values():
    """Experiment with different regularization strengths"""
    X_train, X_test, y_train, y_test = create_dataset()
    lambda_values = [0.001, 0.01, 0.1]
    
    plt.figure(figsize=(15, 5))
    
    for i, lambda_val in enumerate(lambda_values):
        model = create_model_with_l2_regularization(lambda_val)
        model.compile(optimizer=Adam(0.01), loss='binary_crossentropy', metrics=['accuracy'])
        
        history = model.fit(X_train, y_train, 
                           validation_data=(X_test, y_test),
                           epochs=200, batch_size=32, verbose=0)
        
        plt.subplot(1, 3, i+1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title(f'L2 Regularization (λ={lambda_val})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Print final accuracies
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        print(f"λ={lambda_val}: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("=== Regularization in Deep Learning Demo ===\n")
    
    # Main comparison
    compare_regularization_effects()
    
    print("\n=== Experimenting with different λ values ===\n")
    experiment_with_lambda_values()
    
    print("\n=== Key Observations ===")
    print("1. Without regularization: Complex decision boundary, overfits training data")
    print("2. With L2 regularization: Smoother boundary, weights concentrated near zero")
    print("3. With L1 regularization: Creates sparsity (some weights become very small)")
    print("4. Higher λ values lead to stronger regularization (simpler models)")