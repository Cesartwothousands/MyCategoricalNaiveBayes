import numpy as np


def create_imbalanced_dataset(X_train, y_train, alpha_class):
    # Convert inputs to numpy arrays
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    # Make sure X_train and y_train have the same length
    assert len(X_train) == len(y_train)

    # Find unique classes and their counts
    unique_classes, class_counts = np.unique(y_train, return_counts=True)

    # Sample from Dirichlet distribution
    class_ratios = np.random.dirichlet([alpha_class] * len(unique_classes))

    # Compute the number of samples for each class in the imbalanced dataset
    num_samples_per_class = (class_ratios * len(y_train)).astype(int)

    # Ensure at least one sample for each class
    num_samples_per_class = np.maximum(num_samples_per_class, 1)

    # Ensure that we don't request more samples than available for each class
    num_samples_per_class = np.minimum(num_samples_per_class, class_counts)

    # For each class, select the respective number of samples
    X_list = []
    y_list = []
    for class_label, num_samples in zip(unique_classes, num_samples_per_class):
        # Get indices of all samples of this class
        indices = np.where(y_train == class_label)[0]

        # Randomly choose a subset of samples for this class
        chosen_indices = np.random.choice(indices, num_samples, replace=True)

        X_list.append(X_train[chosen_indices])
        y_list.append(y_train[chosen_indices])

    X_imbalanced = np.concatenate(X_list, axis=0)
    y_imbalanced = np.concatenate(y_list, axis=0)

    return X_imbalanced, y_imbalanced