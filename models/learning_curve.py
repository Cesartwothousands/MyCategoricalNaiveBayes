from models.base_estimator import CategoricalNaiveBayes
import matplotlib.pyplot as plt
from sklearn.model_selection import LearningCurveDisplay
import numpy as np

def plot_learning_curve(method, alpha, beta, X_train, y_train):
    """
    Function to plot learning curve for a given method, alpha, and beta values.

    Parameters:
    - method: Estimation method, either "MAP" or "MLE"
    - alpha: Scalar value for alpha
    - beta: Scalar value for beta
    - X_train: Training data
    - y_train: Training labels
    """

    # Initialize the model for the given alpha and beta
    model = CategoricalNaiveBayes(alpha=alpha, beta=beta, method=method)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_prop_cycle('color', ['b', 'r'])  # Set color cycle to blue for training and red for testing

    common_params = {
        "X": X_train,
        "y": y_train,
        "train_sizes": np.linspace(0.1, 1.0, 5),
        "score_type": "both",
        "n_jobs": 4,
        "line_kw": {"marker": "o"},
        "std_display_style": "fill_between",
        "score_name": "Score",
    }

    # Plot learning curve for the model
    LearningCurveDisplay.from_estimator(model, **common_params, ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ["Training Score", "Test Score"])
    ax.set_title(f"Method: {method}, Alpha: {alpha}, Beta: {beta}", fontweight='bold')
    ax.set_xlabel('Training Size', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.grid(True)

    plt.tight_layout()
    plt.show()