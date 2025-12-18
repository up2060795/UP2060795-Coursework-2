# Coursework 2 

PROJECT OVERVIEW
----------------

This project presents an end-to-end data science workflow using the Fashion-MNIST
dataset. The aim is to compare traditional machine learning methods with neural
network approaches and to explore how design choices in neural networks affect
model performance.

The project is structured around three questions:

Q1: A traditional, non-neural-network approach to image classification.
Q2: A neural network approach using a convolutional neural network (CNN).
Q3: An investigation into how the choice of activation function affects CNN
    performance.

----------------------------------------------------------------------------------------

REPOSITORY STRUCTURE
--------------------

.
├── README.txt               # Project overview and instructions
├── dependencies.txt         # Python dependencies and versions
├── functions.py             # Shared helper functions
├── Q1_folder/
│   └── Q1.ipynb             # Traditional ML approach (beginner audience)
├── Q2_folder/
│   └── Q2.ipynb             # CNN approach (beginner audience)
└── Q3_folder/
    ├── Q3_relu.ipynb        # CNN with ReLU activation
    ├── Q3_leaky_relu.ipynb  # CNN with LeakyReLU activation
    └── Q3_sigmoid.ipynb     # CNN with Sigmoid activation


## Dataset Description 

### What
The dataset used in this coursework is the **Fashion-MNIST** dataset.It consists of grayscale images of clothing items, each of size **28×28 pixels**, labelled into **10 different classes** (e.g. T-shirt, trousers, dress, coat).

### Where
The dataset was obtained from **Kaggle**, using the CSV version of Fashion-MNIST. The CSV format allows easy loading with Pandas and is suitable for both traditional machine learning models and convolutional neural networks.

https://www.kaggle.com/datasets/zalando-research/fashionmnist/data

### Why
Fashion-MNIST was chosen because:
- It is simple and suitable for beginners
- It allows fair comparison between classical ML and CNNs
- It is widely used as a benchmark dataset
- It clearly demonstrates the advantages of CNNs on image data

-----------------------------------------------------------------------------------------------------

## Q1 Summary – Decision Tree and Random Forest

Q1 explores classical Random Forest Classifier model

Steps include:
- Data loading and normalisation
- Train/validation split
- Model training and evaluation
- Comparison of validation accuracy
- Discussion of results

Random Forest performs better due to ensemble learning and reduced overfitting.

----------------------------------------------------------------------------------------------------

## Q2 – Convolutional Neural Network with Tanh Activation

In Q2, a convolutional neural network (CNN) is implemented using Tanh as the activation function. The purpose of this notebook is to introduce the structure and training process of CNNs in a clear, tutorial-style manner.

This notebook includes:
- Conversion of image data into PyTorch tensors
- Reshaping images into a 1×28×28 format suitable for convolutional layers
- CNN architecture explanation
- Training loop and loss monitoring
- Validation accuracy evaluation

Tanh is used as a classical activation function to demonstrate CNN learning behaviour, including slower convergence and activation saturation effects.

--------------------------------------------------------------------------------------------------

## Q3 – CNN Activation Function Investigation (ReLU)

In Q3, the CNN architecture from Q2 is reused, but the activation function is changed from Tanh to ReLU, Sigmoid and LeakyReLU. The aim of this section is to investigate how the choice of activation function affects learning speed and model performance.

To ensure a fair comparison, all other aspects of the model remain unchanged:
- Same dataset and preprocessing
- Same train/validation split
- Same CNN architecture
- Same optimiser and number of training epochs

Results show that:

- Sigmoid suffers from vanishing gradients and fails to learn effectively

- ReLU provides a strong baseline performance

- LeakyReLU improves training stability and convergence

This highlights the importance of activation function choice in CNNs.

--------------------------------------------------------------------------------------------------

### Data Cleaning and Normalisation

The Fashion-MNIST dataset used in this coursework is pre-cleaned and does not contain missing or invalid values. As a result, no additional data cleaning steps were required.

Pixel values were normalised from the range 0–255 to 0–1 prior to training. This normalisation improves numerical stability, helps gradient-based optimisation, and ensures consistent model performance across Q1, Q2, and Q3.

----------------------------------------------------------------------------------------------------

## Features and Preprocessing

Each Fashion-MNIST image is represented by its raw pixel intensities.
All 784 pixels (28×28 grayscale image) are used as input features.

Pixel values are normalised to the range [0, 1] to improve numerical stability
and ensure consistent feature scaling.

No additional handcrafted features are created.
This design choice allows for a fair comparison between traditional machine learning
methods (Question 1) and neural network approaches (Questions 2 and 3),
where feature learning is performed automatically by the model.


----------------------------------------

##  Dependencies & Reproducibility

All experiments were run using the following versions:

python==3.9.13
numpy==1.23.4
pandas==1.5.1
scikit-learn==<verified_version>
matplotlib==3.6.2
torch==1.13.0
torchvision==0.14.0
seaborn==0.13.2

---------------------------------

##  How to Run the Project

Clone the repository

Install dependencies using:
pip install -r dependencies.txt

Run notebooks in order:

Q1_folder/Q1.ipynb

Q2_folder/Q2.ipynb

Q3_folder/*.ipynb

----------------------------
NOTES ON EXTERNAL RESOURCES
-------------------------------------------------

This project was completed using a combination of original code and ideas informed by external learning resources.

Conceptual understanding and implementation guidance were drawn from:
- University lecture materials and class notes
- Official PyTorch documentation
- Scikit-learn documentation
- Online programming discussions (e.g. Stack Overflow)
- Kaggle notebooks and dataset documentation
- ChatGPT, used as a learning and debugging aid

All code in this repository was written and adapted by me. No external code was copied directly.
