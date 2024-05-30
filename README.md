# Cats and Dogs Classifier

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![GitHub issues](https://img.shields.io/github/issues/yourusername/cats-and-dogs-classifier)
![GitHub stars](https://img.shields.io/github/stars/yourusername/cats-and-dogs-classifier)
![GitHub license](https://img.shields.io/github/license/yourusername/cats-and-dogs-classifier)

## Overview

The Cats and Dogs Classifier is a deep learning project that uses a convolutional neural network (CNN) to classify images of cats and dogs. This project leverages TensorFlow and Keras for building and training the model, along with data augmentation techniques to improve the model's robustness.

## Features

- **Data Augmentation**: Techniques like rotation, shifting, and flipping to increase the diversity of the training data.
- **Convolutional Neural Network**: A deep learning model with multiple convolutional layers for feature extraction.
- **Model Evaluation**: Visualizations of training and validation accuracy and loss.
- **Pre-trained Model**: Save and load the trained model for future use.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/cats-and-dogs-classifier.git
    ```
2. Change to the project directory:
    ```bash
    cd cats-and-dogs-classifier
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Download the dataset:
    ```python
    from tensorflow.keras.utils import get_file

    url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    path_to_zip = get_file('cats_and_dogs.zip', origin=url, extract=True)
    ```
2. Set paths for training and validation data:
    ```python
    import os

    dataset_path = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
    train_dir = os.path.join(dataset_path, 'train')
    validation_dir = os.path.join(dataset_path, 'validation')
    ```
3. Run the training script to train the model:
    ```python
    python train.py
    ```
4. Evaluate the model and visualize the results:
    ```python
    python evaluate.py
    ```

## Dataset

The dataset used in this project is the Cats and Dogs dataset from Microsoft. It contains 25,000 images of cats and dogs (12,500 images per class) for training and validation.

## Model Architecture

The CNN model consists of:
- Four convolutional layers with ReLU activation and max pooling.
- A flatten layer to convert the 2D matrix data to a vector.
- Two dense (fully connected) layers with ReLU activation.
- A dropout layer to prevent overfitting.
- An output layer with a sigmoid activation function for binary classification.

## Results

The training and validation accuracy and loss are plotted for each epoch to visualize the model's performance.

![Training and validation accuracy](training_validation_accuracy.png)
![Training and validation loss](training_validation_loss.png)

The trained model is saved as `cats_and_dogs_classifier.h5`.

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries or issues, please contact:

- **Name**: Chandrashekhar K S
- **Email**: cg4618651@gmail.com
- **GitHub**: [ChandrashekharkS](https://github.com/ChandrashekharkS)
