from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from models.load_data import load_data


class CategoricalNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1, beta=1, method="MLE"):
        self.alpha = alpha  # hyperparameter for MAP estimation of class probabilities
        self.beta = beta  # hyperparameter for MAP estimation of pixel probabilities
        self.method = method  # Estimation method: "MLE" or "MAP"

        # Initialization of class related attributes
        self.classes_ = None  # A list of unique classes in the dataset
        self.class_probs_ = None  # List containing probabilities of each class
        self.pixel_probs_ = None  # Dictionary with classes as keys and pixel probabilities as values

        # Internal variable to store probabilities from the latest prediction
        self._latest_probs = None

    # Task 1: Build a Fit Method using MLE and MAP for model parameters
    def fit(self, X, y):
        self.classes_ = np.unique(y)

        # Estimate class probabilities
        class_counts = np.array([np.sum(y == c) for c in self.classes_])
        total_samples = len(y)

        if self.method == "MLE":
            self.class_probs_ = class_counts / total_samples
        elif self.method == "MAP":
            self.class_probs_ = (class_counts + self.alpha - 1) / (total_samples + len(self.classes_) * self.alpha)

        # Estimate pixel probabilities
        if self.method == "MLE":
            self.class_probs_ = [np.mean(y == c) for c in self.classes_]
            self.pixel_probs_ = {c: np.mean(X[y == c], axis=0) for c in self.classes_}
        elif self.method == "MAP":
            self.class_probs_ = [(np.sum(y == c) + self.alpha - 1) / (len(y) + len(self.classes_) * (self.alpha - 1))
                                 for c in self.classes_]
            self.pixel_probs_ = {c: (np.sum(X[y == c], axis=0) + self.beta - 1) / (np.sum(y == c) + 2 * (self.beta - 1))
                                 for c in self.classes_}

    # Task 2: Build a Predict Method using MLE and MAP for model parameters
    def predict(self, X):
        epsilon = 1e-10  # Small constant to prevent division by zero
        min_prob = 1e-200  # Threshold to prevent very small probability product
        all_probs = []

        for x in X:
            probs = []
            for c in self.classes_:
                prob = self.class_probs_[self.classes_.tolist().index(c)]
                for idx, pixel in enumerate(x):
                    prob *= self.pixel_probs_[c][idx] * pixel + (1 - self.pixel_probs_[c][idx]) * (1 - pixel)
                    prob = max(prob, min_prob)  # Ensure the probability does not get too small
                probs.append(prob)

            probs_sum = sum(probs)
            if probs_sum <= epsilon:  # Avoid division by zero or a very small number
                probs_sum += epsilon
            normalized_probs = [p / probs_sum for p in probs]
            all_probs.append(normalized_probs)

        self._latest_probs = all_probs
        return [self.classes_[np.argmax(probs)] for probs in all_probs]

    # Task 3: Build a Score Method using MLE and MAP for model parameters
    def score(self, X, y):
        # Make predictions to get the latest probabilities
        self.predict(X)

        # Compute the average log likelihood of the data
        log_likelihoods = [np.log(max(prob)) for prob in self._latest_probs]
        avg_log_likelihood = np.mean(log_likelihoods)

        return avg_log_likelihood


# To load 'balanced' data:
X_train, y_train = load_data('balanced', 'train')
X_test, y_test = load_data('balanced', 'test')
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# Initialize the Naive Bayes Classifier
clf_mle = CategoricalNaiveBayes(method="MLE")
clf_map = CategoricalNaiveBayes(method="MAP")

# Fit the model
clf_mle.fit(X_train, y_train)
clf_map.fit(X_train, y_train)

# Predict and score on a subset of the data for testing
for n_samples in [10, 100, 250]:
    X_subset = X_test[:n_samples]
    y_subset = y_test[:n_samples]
    # print(f"Subset {X_subset}")
    # print(f"Subset {y_subset}")
    predictions_mle = clf_mle.predict(X_subset)
    predictions_map = clf_map.predict(X_subset)
    score_mle = clf_mle.score(X_subset, y_subset)
    score_map = clf_map.score(X_subset, y_subset)

    # print(f"Sample of subset: {y_subset}")
    # print(f"Predictions of subset: {predictions_mle}")
    print(f"Score for MLE {n_samples} samples: {score_mle:.4f}")
    print(f"Score for MAP {n_samples} samples: {score_map:.4f}")