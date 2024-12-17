import numpy as np
from helpers import load_csv_data, create_csv_submission
from improved_logistic import (
    improved_logistic_regression, 
    preprocess_data, 
    predict, 
    compute_balanced_accuracy
)

def main():
    print("Loading data...")
    # Load the data - note we assume the CSV files are in the 'data' directory
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/")
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Training labels shape: {y_train.shape}")

    print("\nClass distribution in training set:")
    print(f"Class -1: {np.sum(y_train == -1)}")
    print(f"Class 1: {np.sum(y_train == 1)}")
    print(f"Ratio (positive/negative): {np.sum(y_train == 1) / np.sum(y_train == -1):.3f}")

    # Preprocess data
    print("\nPreprocessing data...")
    x_train_processed, y_train = preprocess_data(x_train, y_train)
    x_test_processed, _ = preprocess_data(x_test, None)

    print(f"Processed training data shape: {x_train_processed.shape}")
    
    # Initialize parameters for logistic regression
    initial_w = np.zeros(x_train_processed.shape[1])
    max_iters = 200 # more iterations (prev 100)
    gamma = 0.01 # adjusted learning rate (prev 0.01)
    
    # Split some data for validation
    np.random.seed(1)
    indices = np.random.permutation(len(x_train_processed))
    val_size = int(0.2 * len(x_train_processed))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    x_val = x_train_processed[val_indices]
    y_val = y_train[val_indices]
    x_train_final = x_train_processed[train_indices]
    y_train_final = y_train[train_indices]
    
    print("\nTraining model...")
    w, loss = improved_logistic_regression(
        y_train_final, 
        x_train_final, 
        initial_w, 
        max_iters, 
        gamma
    )
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_predictions = predict(x_val, w)
    balanced_acc = compute_balanced_accuracy(y_val, val_predictions)
    print(f"Balanced accuracy on validation set: {balanced_acc:.3f}")
    
    print("\nMaking predictions on test set...")
    y_pred = predict(x_test_processed, w)
    
    # Check prediction distribution
    print("\nPrediction distribution:")
    print(f"Class -1: {np.sum(y_pred == -1)}")
    print(f"Class 1: {np.sum(y_pred == 1)}")
    
    print("\nCreating submission file...")
    create_csv_submission(test_ids, y_pred, "predictions.csv")
    print("Done! Submission file 'predictions.csv' has been created.")

if __name__ == "__main__":
    main()