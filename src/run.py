import numpy as np
from helpers import load_csv_data, create_csv_submission
from implementations import logistic_regression  # You'll need to import your implementations

def main():
    print("Loading data...")
    # Load the data - note we assume the CSV files are in the 'data' directory
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data("data/")
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Training labels shape: {y_train.shape}")
    
    # Initialize parameters for logistic regression
    initial_w = np.zeros(x_train.shape[1])
    max_iters = 100
    gamma = 0.01
    
    print("\nTraining model...")
    # Train the model
    w, loss = logistic_regression(y_train, x_train, initial_w, max_iters, gamma)
    
    print("Making predictions...")
    # Make predictions on test set
    y_pred = predict(x_test, w)
    
    print("Creating submission file...")
    # Create submission file
    create_csv_submission(test_ids, y_pred, "predictions.csv")
    print("Done! Submission file 'predictions.csv' has been created.")

def predict(x, w):
    """Make predictions using trained weights"""
    # Convert probabilities to -1/1 predictions
    prob = 1 / (1 + np.exp(-x.dot(w)))
    return (2 * (prob >= 0.5) - 1).astype(int)  # Convert to -1/1

if __name__ == "__main__":
    main()