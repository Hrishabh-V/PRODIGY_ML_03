
# Cat vs Dog Image Classifier using SVM

This repository contains an image classification project that distinguishes between cats and dogs using a Support Vector Machine (SVM). The project includes a pre-trained SVM model and a simple graphical interface that allows users to upload images and receive predictions.

## Introduction

Support Vector Machines (SVM) are supervised learning models used for classification and regression tasks. In this project, we leverage an SVM to classify images of cats and dogs based on their pixel values.

SVM works by finding the optimal hyperplane that best separates the data into two distinct classes. It maximizes the margin between the closest points from both classes (known as support vectors), ensuring a clear separation of the categories (cats and dogs in this case). In higher dimensions, the SVM effectively handles complex classification problems by using kernel functions to transform data.

## Project Structure

- **SVM.ipynb**: A Jupyter Notebook that preprocesses the data, trains the SVM model, and saves it for use in the application.
- **app.py**: A GUI-based application where users can upload an image, and the SVM model predicts whether the image is of a cat or a dog.
- **svm_model.pkl**: The pre-trained SVM model used in the application.


## Getting Started

### Prerequisites

To get started, you will need to install the following dependencies:

```bash
pip install numpy opencv-python scikit-learn customtkinter
```

### How SVM Works

SVM is a powerful algorithm that constructs a hyperplane or set of hyperplanes in a high-dimensional space. In this project:
- The input images are resized to 128x128 pixels, flattened, and normalized.
- SVM classifies each image by determining which side of the hyperplane it falls on, assigning a label of either "cat" or "dog".
- The model is trained on a dataset of 4,000 cat and 4,000 dog images, learning the pixel patterns that distinguish between the two.
- The dataset used for training the model can be found [here](https://www.kaggle.com/c/dogs-vs-cats/data).



By maximizing the margin between the support vectors of each class, SVM ensures a robust and accurate separation, even in cases where the data is not linearly separable. The margin is defined as the distance between the hyperplane and the nearest data points of each class.

## Usage


### Run the Application

To classify an image using the pre-trained SVM model:

1. Run the `app.py` file:

    ```bash
    python app.py
    ```

2. Upload an image via the user interface, and the model will predict whether it's a cat or a dog.

## Example

1. After training, the model can distinguish between cats and dogs with high accuracy.
2. The application provides real-time feedback on image classification, making it easy to test new images.

## Future Enhancements

- **Improved Classification**: Use Convolutional Neural Networks (CNNs) for higher accuracy in image classification.
- **Additional Classes**: Extend the project to recognize other animals.
- **Real-time Classification**: Enhance the app for real-time image classification using webcam inputs.


unzip the model first


