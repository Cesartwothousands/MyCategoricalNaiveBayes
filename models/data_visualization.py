import idx2numpy
import matplotlib.pyplot as plt
import os
import numpy as np

# initialize the data directory
data_dir = r'C:\aCesar\F_File\MSCS\CS535_Machie Learning I\mini-project\EMNIST-Naive-Bayes\dataset\EMNIST\raw'
pdf_output_path = r'C:\aCesar\F_File\MSCS\CS535_Machie Learning I\mini-project\EMNIST-Naive-Bayes\dataset\EMNIST\pdf_output'

# map the file names and their corresponding labels
datasets = {
    'balanced-test': ('emnist-balanced-test-images-idx3-ubyte', 'emnist-balanced-test-labels-idx1-ubyte'),
    'balanced-train': ('emnist-balanced-train-images-idx3-ubyte', 'emnist-balanced-train-labels-idx1-ubyte'),
    'byclass-test': ('emnist-byclass-test-images-idx3-ubyte', 'emnist-byclass-test-labels-idx1-ubyte'),
    'byclass-train': ('emnist-byclass-train-images-idx3-ubyte', 'emnist-byclass-train-labels-idx1-ubyte'),
    'bymerge-test': ('emnist-bymerge-test-images-idx3-ubyte', 'emnist-bymerge-test-labels-idx1-ubyte'),
    'bymerge-train': ('emnist-bymerge-train-images-idx3-ubyte', 'emnist-bymerge-train-labels-idx1-ubyte'),
    'digits-test': ('emnist-digits-test-images-idx3-ubyte', 'emnist-digits-test-labels-idx1-ubyte'),
    'digits-train': ('emnist-digits-train-images-idx3-ubyte', 'emnist-digits-train-labels-idx1-ubyte'),
    'letters-test': ('emnist-letters-test-images-idx3-ubyte', 'emnist-letters-test-labels-idx1-ubyte'),
    'letters-train': ('emnist-letters-train-images-idx3-ubyte', 'emnist-letters-train-labels-idx1-ubyte'),
    'mnist-test': ('emnist-mnist-test-images-idx3-ubyte', 'emnist-mnist-test-labels-idx1-ubyte'),
    'mnist-train': ('emnist-mnist-train-images-idx3-ubyte', 'emnist-mnist-train-labels-idx1-ubyte'),
}


def visualize_dataset(image_file, label_file):
    # Convert the idx files to numpy arrays
    images = idx2numpy.convert_from_file(image_file)
    labels = idx2numpy.convert_from_file(label_file)

    # Define the number of samples and character IDs
    num_samples_per_char = 5
    unique_char_ids = np.unique(labels)

    # Initialize the figure for visualization
    figure = plt.figure(figsize=(10, 10))
    cols = len(unique_char_ids)
    rows = num_samples_per_char

    for j, char_id in enumerate(unique_char_ids):
        idxs = np.where(labels == char_id)[0]
        random_idxs = np.random.choice(idxs, num_samples_per_char, replace=False)

        for i in range(num_samples_per_char):
            ax = figure.add_subplot(rows, cols, i * cols + j + 1)

            # First, flip the image horizontally and then rotate it by 90 degrees
            img = np.rot90(np.fliplr(images[random_idxs[i]]))

            if i == 0:
                ax.set_title(str(char_id), fontsize=18)
            ax.imshow(img, cmap=plt.cm.binary)
            ax.axis("off")

    # Adjust layout and save the visualization as a PDF
    plt.tight_layout()
    plt.savefig(os.path.join(pdf_output_path, f"emnist-{os.path.basename(image_file)}.pdf"), bbox_inches='tight')
    plt.show()


for dataset_name, (img_file, lbl_file) in datasets.items():
    print(f"Visualizing {dataset_name}...")
    img_path = os.path.join(data_dir, img_file)
    lbl_path = os.path.join(data_dir, lbl_file)
    visualize_dataset(img_path, lbl_path)