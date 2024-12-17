import numpy as np

def compute_balanced_accuracy(y_true, y_pred):
    """Compute balanced accuracy to handle class imbalance"""
    positives = (y_true == 1)
    negatives = (y_true == -1)
    
    true_pos = np.sum((y_pred == 1) & positives)
    true_neg = np.sum((y_pred == -1) & negatives)
    
    sensitivity = true_pos / np.sum(positives) if np.sum(positives) > 0 else 0
    specificity = true_neg / np.sum(negatives) if np.sum(negatives) > 0 else 0
    
    return (sensitivity + specificity) / 2

def preprocess_data(x, y):
    """Preprocess data with better handling of class imbalance"""
    # 1. Handle missing values
    x = handle_missing_values(x)
    
    # 2. Feature scaling
    x = standardize_features(x)
    
    # 3. Add polynomial features
    x = add_polynomial_features(x)
    
    return x, y

def handle_missing_values(x):
    """Handle missing values in features"""
    # Replace special codes with NaN
    special_codes = [77, 88, 99, 9999]
    for code in special_codes:
        x[x == code] = np.nan
    
    # Fill NaN with median of each column
    for j in range(x.shape[1]):
        mask = np.isnan(x[:, j])
        if mask.any():
            median_val = np.nanmedian(x[:, j])
            x[mask, j] = median_val
    
    return x

def standardize_features(x):
    """Standardize features to have zero mean and unit variance"""
    means = np.mean(x, axis=0)
    stds = np.std(x, axis=0)
    stds[stds == 0] = 1  # Prevent division by zero
    return (x - means) / stds

def add_polynomial_features(x):
    """Add polynomial features (squared terms) for potentially better separation"""
    # Add squared terms for numeric features
    x_poly = np.column_stack([x, x**2])
    return x_poly

def improved_logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Improved logistic regression implementation
    - Uses class weights to handle imbalance
    - Better initialization
    - Adaptive learning rate
    """
    # Compute class weights
    n_samples = len(y)
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == -1)
    
    # Adjust weights for class imbalance
    pos_weight = n_samples / (2 * n_pos)
    neg_weight = n_samples / (2 * n_neg)
    sample_weights = np.where(y == 1, pos_weight, neg_weight)
    
    # Initialize parameters
    w = initial_w
    y_binary = (y + 1) / 2  # Convert labels from {-1,1} to {0,1}
    
    # Training loop
    for iter in range(max_iters):
        # Compute predictions
        tx_w = tx.dot(w)
        pred = sigmoid(tx_w)
        
        # Compute weighted gradient
        error = pred - y_binary
        gradient = tx.T.dot(error * sample_weights) / n_samples
        
        # Adaptive learning rate
        gamma_t = gamma / np.sqrt(iter + 1)
        
        # Update weights
        w = w - gamma_t * gradient
        
        # Optional: Early stopping if gradient is small
        if np.abs(gradient).max() < 1e-7:
            break
    
    # Compute final loss
    pred = sigmoid(tx.dot(w))
    loss = compute_loss(y_binary, pred)
    
    return w, loss

def sigmoid(t):
    """Numerically stable sigmoid function"""
    t = np.clip(t, -700, 700)  # Prevent overflow
    return 1.0 / (1.0 + np.exp(-t))

def compute_loss(y, pred):
    """Compute binary cross-entropy loss"""
    epsilon = 1e-15  # Small constant to prevent log(0)
    pred = np.clip(pred, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))

def predict_proba(x, w):
    """Predict probabilities"""
    return sigmoid(x.dot(w))

def predict(x, w, threshold=0.5):
    """Make predictions with a custom threshold"""
    probas = predict_proba(x, w)
    return 2 * (probas >= threshold) - 1  # Convert to {-1,1}