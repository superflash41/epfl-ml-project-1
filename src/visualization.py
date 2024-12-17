import numpy as np
import matplotlib.pyplot as plt
from helpers import load_csv_data
from improved_logistic import preprocess_data, improved_logistic_regression, predict, compute_balanced_accuracy

def plot_class_distribution(y_train):
    """Plot the distribution of classes in training data"""
    plt.figure(figsize=(8, 6))
    classes, counts = np.unique(y_train, return_counts=True)
    plt.bar(['Negative (-1)', 'Positive (1)'], counts)
    plt.title('Class Distribution in Training Data')
    plt.ylabel('Number of Samples')
    plt.savefig('report/figures/class_distribution.png')
    plt.close()

def plot_feature_importance(x_train, w, feature_names=None):
    """Plot the importance of each feature based on model weights"""
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(w))]
    
    # Get absolute weights and sort them
    importance = np.abs(w)
    sorted_idx = np.argsort(importance)
    pos = np.arange(sorted_idx[-20:].shape[0]) + .5

    plt.figure(figsize=(10, 6))
    plt.barh(pos, importance[sorted_idx[-20:]])
    plt.yticks(pos, np.array(feature_names)[sorted_idx[-20:]])
    plt.title('Top 20 Most Important Features')
    plt.xlabel('Absolute Weight Value')
    plt.tight_layout()
    plt.savefig('report/figures/feature_importance.png')
    plt.close()

def plot_learning_curves(x_train, y_train):
    """Plot learning curves with different training set sizes"""
    train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9]
    train_scores = []
    val_scores = []

    # Split validation set
    np.random.seed(1)
    indices = np.random.permutation(len(x_train))
    val_size = int(0.2 * len(x_train))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    x_val = x_train[val_indices]
    y_val = y_train[val_indices]
    
    for size in train_sizes:
        # Use subset of training data
        subset_size = int(len(train_indices) * size)
        subset_indices = train_indices[:subset_size]
        
        x_train_subset = x_train[subset_indices]
        y_train_subset = y_train[subset_indices]
        
        # Train model
        initial_w = np.zeros(x_train_subset.shape[1])
        w, _ = improved_logistic_regression(
            y_train_subset, x_train_subset, initial_w, max_iters=100, gamma=0.01
        )
        
        # Compute scores
        train_pred = predict(x_train_subset, w)
        val_pred = predict(x_val, w)
        
        train_score = compute_balanced_accuracy(y_train_subset, train_pred)
        val_score = compute_balanced_accuracy(y_val, val_pred)
        
        train_scores.append(train_score)
        val_scores.append(val_score)

    plt.figure(figsize=(8, 6))
    plt.plot([size * 100 for size in train_sizes], train_scores, 'o-', label='Training Score')
    plt.plot([size * 100 for size in train_sizes], val_scores, 'o-', label='Validation Score')
    plt.xlabel('Training Set Size (%)')
    plt.ylabel('Balanced Accuracy')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('report/figures/learning_curves.png')
    plt.close()

def plot_hyperparameter_tuning(x_train, y_train):
    """Plot model performance with different hyperparameters"""
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    accuracies = []

    # Split validation set
    np.random.seed(1)
    indices = np.random.permutation(len(x_train))
    val_size = int(0.2 * len(x_train))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    x_val = x_train[val_indices]
    y_val = y_train[val_indices]
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]

    for gamma in learning_rates:
        initial_w = np.zeros(x_train.shape[1])
        w, _ = improved_logistic_regression(
            y_train, x_train, initial_w, max_iters=100, gamma=gamma
        )
        
        val_pred = predict(x_val, w)
        accuracy = compute_balanced_accuracy(y_val, val_pred)
        accuracies.append(accuracy)

    plt.figure(figsize=(8, 6))
    plt.semilogx(learning_rates, accuracies, 'o-')
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Accuracy')
    plt.title('Model Performance vs Learning Rate')
    plt.grid(True)
    plt.savefig('report/figures/hyperparameter_tuning.png')
    plt.close()

def main():
    # Load and preprocess data
    print("Loading data...")
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/")
    x_train_processed, y_train = preprocess_data(x_train, y_train)
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Ensure the figures directory exists
    import os
    os.makedirs('report/figures', exist_ok=True)
    
    plot_class_distribution(y_train)
    plot_learning_curves(x_train_processed, y_train)
    plot_hyperparameter_tuning(x_train_processed, y_train)
    
    # Train final model to get feature importance
    initial_w = np.zeros(x_train_processed.shape[1])
    w, _ = improved_logistic_regression(
        y_train, x_train_processed, initial_w, max_iters=100, gamma=0.01
    )
    plot_feature_importance(x_train_processed, w)
    
    print("Visualizations have been saved in the report/figures directory")

if __name__ == "__main__":
    main()