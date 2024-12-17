import numpy as np

def compute_mse_loss(y, tx, w):
    """Calculate MSE loss.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w: numpy array of shape (D,). The vector of model parameters.
    Returns:
        float: MSE loss
    """
    e = y - tx.dot(w)
    return 1/(2*len(y)) * e.dot(e)

def compute_gradient(y, tx, w):
    """Compute gradient of MSE loss.
    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N,D)
        w: numpy array of shape (D,)
    Returns:
        numpy array of shape (D,): The gradient of MSE loss
    """
    e = y - tx.dot(w)
    return -1/len(y) * tx.T.dot(e)

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.
    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N,D)
        initial_w: numpy array of shape (D,). The initial guess for model parameters
        max_iters: int. The number of iterations to run
        gamma: float. The step size
    Returns:
        w: numpy array of shape (D,). The last weight vector
        loss: float. The corresponding loss value (cost function)
    """
    w = initial_w
    for _ in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient
    loss = compute_mse_loss(y, tx, w)
    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent.
    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N,D)
        initial_w: numpy array of shape (D,). The initial guess for model parameters
        max_iters: int. The number of iterations to run
        gamma: float. The step size
    Returns:
        w: numpy array of shape (D,). The last weight vector
        loss: float. The corresponding loss value (cost function)
    """
    w = initial_w
    for _ in range(max_iters):
        # Select random data point
        i = np.random.randint(len(y))
        gradient = -1 * tx[i:i+1].T.dot(y[i:i+1] - tx[i:i+1].dot(w))
        w = w - gamma * gradient
    loss = compute_mse_loss(y, tx, w)
    return w, loss

def least_squares(y, tx):
    """Least squares regression using normal equations.
    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N,D)
    Returns:
        w: numpy array of shape (D,). The optimal weights
        loss: float. The corresponding loss value (cost function)
    """
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_mse_loss(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations.
    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N,D)
        lambda_: float. The regularization parameter
    Returns:
        w: numpy array of shape (D,). The optimal weights
        loss: float. The corresponding loss value (cost function)
    """
    aI = 2 * len(y) * lambda_ * np.identity(tx.shape[1])
    w = np.linalg.solve(tx.T.dot(tx) + aI, tx.T.dot(y))
    loss = compute_mse_loss(y, tx, w)  # Return MSE without regularization term
    return w, loss

def sigmoid(t):
    """Apply sigmoid function"""
    return 1.0 / (1 + np.exp(-t))

def compute_logistic_loss(y, tx, w):
    """Compute negative log likelihood loss for logistic regression"""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred + 1e-15)) + (1 - y).T.dot(np.log(1 - pred + 1e-15))
    return -loss/len(y)

def compute_logistic_gradient(y, tx, w):
    """Compute gradient of negative log likelihood loss for logistic regression"""
    pred = sigmoid(tx.dot(w))
    return tx.T.dot(pred - y) / len(y)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent.
    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N,D)
        initial_w: numpy array of shape (D,). The initial guess for model parameters
        max_iters: int. The number of iterations to run
        gamma: float. The step size
    Returns:
        w: numpy array of shape (D,). The last weight vector
        loss: float. The corresponding loss value (cost function)
    """
    w = initial_w
    for _ in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w)
        w = w - gamma * gradient
    loss = compute_logistic_loss(y, tx, w)
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent.
    Args:
        y: numpy array of shape (N,)
        tx: numpy array of shape (N,D)
        lambda_: float. The regularization parameter
        initial_w: numpy array of shape (D,). The initial guess for model parameters
        max_iters: int. The number of iterations to run
        gamma: float. The step size
    Returns:
        w: numpy array of shape (D,). The last weight vector
        loss: float. The corresponding loss value (cost function)
    """
    w = initial_w
    for _ in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient
    loss = compute_logistic_loss(y, tx, w)  # Return loss without regularization term
    return w, loss