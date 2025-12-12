import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ---------------------------------------------------
# 1. Load Fashion-MNIST CSV
# ---------------------------------------------------
def get_data(csv_path="data/fashion-mnist_test.csv", normalize=True):
    # Step 1: read CSV into pandas dataframe
    df = pd.read_csv(csv_path)

    # Step 2: convert dataframe to numpy array
    data_array = df.to_numpy()

    # Step 3: first column is labels, rest are pixels
    y = data_array[:, 0]          # labels
    X = data_array[:, 1:].astype("float32")   # pixels

    # Step 4: optional normalisation
    if normalize:
        X = X / 255.0

    # Step 5: return numpy arrays
    return X, y

# ---------------------------------------------------
# 2. Split into train/val sets
# ---------------------------------------------------
def data_split_train_val(X, y, test_size=0.2, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    return X_train, X_val, y_train, y_val


# ---------------------------------------------------
# 3. Reshape flat pixels into 28x28 images (for PyTorch/CNNs)
# ---------------------------------------------------
def reshape_images(X):
    """
    Reshapes a (N, 784) array into (N, 1, 28, 28) for CNNs.
    """
    return X.reshape(-1, 1, 28, 28)


# ---------------------------------------------------
# 4. Visualise sample images
# ---------------------------------------------------
def show_samples(X, y, n=9):
    plt.figure(figsize=(6, 6))
    for i in range(n):
        img = X[i].reshape(28, 28)
        plt.subplot(3, 3, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {y[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
