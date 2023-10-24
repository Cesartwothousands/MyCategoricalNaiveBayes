import idx2numpy
import os


def load_data(dataset_type='balanced', data_type='train'):
    # Define the file paths based on dataset_type and data_type
    images_file = f"C:\\aCesar\\F_File\\MSCS\\CS535_Machie Learning I\\mini-project\\data\\EMNIST\\raw\\emnist-{dataset_type}-{data_type}-images-idx3-ubyte"
    labels_file = f"C:\\aCesar\\F_File\\MSCS\\CS535_Machie Learning I\\mini-project\\data\\EMNIST\\raw\\emnist-{dataset_type}-{data_type}-labels-idx1-ubyte"

    # Ensure files exist
    if not os.path.exists(images_file) or not os.path.exists(labels_file):
        raise ValueError(f"Files related to {dataset_type} and {data_type} not found!")

    # Convert idx files to numpy arrays
    X = idx2numpy.convert_from_file(images_file)
    y = idx2numpy.convert_from_file(labels_file)
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    # print(X.shape, y.shape)

    return X, y