import numpy as np
import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from sklearn.model_selection import train_test_split


def get_data(data, indices=None, binarize=True):
    N = len(data)
    if indices is None:
        indices = range(0, N)
    X = np.stack([data[i][0].numpy() for i in indices], axis=0)  # (N,28,28)
    if binarize:
        X = (X > 0.5)
    y = np.array([data[i][1] for i in indices])
    return X, y


def main():
    # Define data transformations
    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()])

    # Load the dataset
    data = datasets.EMNIST(
        root=r"C:\aCesar\F_File\MSCS\CS535_Machie Learning I\mini-project\EMNIST-Naive-Bayes\dataset",
        split="balanced",
        download=True,
        transform=data_transform
    )

    # Use the function to extract data
    X, y = get_data(data)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.90, random_state=0)

    # (You can add further code here for modeling or other tasks with the dataset)


if __name__ == '__main__':
    main()
