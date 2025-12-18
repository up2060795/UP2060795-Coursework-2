import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ---------------------------------------------------
# 1. Load Fashion-MNIST CSV
# ---------------------------------------------------
def get_data(csv_path="data/fashion-mnist_test.csv", normalize=True):
    """
    Load Fashion-MNIST data and return features and labels.
    Optionally normalise pixel values to [0, 1].
    """
    df = pd.read_csv(csv_path)

    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.astype("float32")

    if normalize:
        X = X / 255.0

    return X, y


# ---------------------------------------------------
# 2. Split into train/validation sets
# ---------------------------------------------------
def data_split_train_val(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and validation sets using stratification.
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


# ---------------------------------------------------
# 3. Reshape flat pixels into images (for CNNs)
# ---------------------------------------------------
def reshape_images(X):
    """
    Reshape (N, 784) array into (N, 1, 28, 28) for CNN input.
    """
    return X.reshape(-1, 1, 28, 28)


# ---------------------------------------------------
# 4. Visualise sample images
# ---------------------------------------------------
def show_samples(X, y, n=9, class_names=None):
    """
    Display n random Fashion-MNIST images with labels.
    """
    plt.figure(figsize=(6, 6))
    indices = np.random.choice(len(X), n, replace=False)

    for i, idx in enumerate(indices):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X[idx].reshape(28, 28), cmap="gray")

        label = y[idx]
        if class_names:
            label = class_names[label]

        plt.title(f"{label}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
