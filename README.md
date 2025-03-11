# Fruit_CNN

This project is a binary classification model that uses Convolutional Neural Networks (CNN) to classify fruits as either fresh or rotten. The goal of this project is to develop a machine learning model that can distinguish between fresh and rotten apples, bananas and oranges based on images (in code we predict only one type of fruit images). It was my first CNN model that I created during the first semester of my master's degree.

## Project Overview

In this project, we use CNN to train a model on images of apples categorized into two classes:

- **FreshApple (Class 1)**: Healthy apples.
- **RottenApple (Class 0)**: Rotten apples.

The model takes images as input and classifies them into one of the two classes, fresh or rotten. The dataset consists of images stored in separate directories for fresh and rotten apples, and a test set with labeled images.

## Dataset

The dataset used in this project consists of images of apples, categorized into two main folders:

- **FreshApple**: Images of healthy, fresh apples.
- **RottenApple**: Images of spoiled, rotten apples.

The dataset is divided into training and testing subsets:

- **Training Data**: Images used to train the CNN model.
- **Testing Data**: Images used to evaluate the model's performance.

## Model Architecture

The model consists of the following layers:

- **Convolutional Layer (Conv2D)**: Applies filters to the image to extract features.
- **Max Pooling Layer (MaxPooling2D)**: Reduces the spatial dimensions of the feature maps.
- **Flatten Layer**: Converts the 2D feature maps into a 1D vector.
- **Fully Connected Layer (Dense)**: Performs classification by using the extracted features.
- **Output Layer**: A sigmoid activation function is used to predict the probability of an image being fresh or rotten.

## Libraries and Tools Used

- **Python**: The main programming language used for this project.
- **OpenCV**: For image loading and processing.
- **TensorFlow/Keras**: For building and training the CNN model.
- **NumPy**: For numerical computations.
- **Matplotlib**: For plotting loss and accuracy graphs and the ROC curve.
- **scikit-learn**: For metrics like ROC curve and confusion matrix.

## Issues and Improvements

- **Class Imbalance**: The model might struggle if the test set has an unequal distribution of the two classes.
- **Overfitting**: The model could potentially overfit on the training data. Regularization techniques such as dropout or data augmentation might help improve generalization.
- **Model Performance**: Further hyperparameter tuning or using more advanced architectures like ResNet or VGG could improve classification performance.

## Acknowledgments

- **OpenCV** for image loading and resizing.
- **TensorFlow/Keras** for providing the tools to build and train neural networks.
- **Matplotlib and scikit-learn** for generating evaluation plots and metrics.

## Results

After training the model for 50 epochs, the model will provide:

- **Test accuracy**: The percentage of correctly classified images in the test set.
- 
![accuracy](https://github.com/user-attachments/assets/ff7e8c90-e57a-4af8-bb5b-4b7b8718b6f1)

- **Confusion Matrix**: To evaluate the number of true positives, false positives, true negatives, and false negatives.
- 
![conf_matrix](https://github.com/user-attachments/assets/d36fe314-92e9-4bb2-a87d-e3067e5c81d0)

- **ROC Curve**: The Receiver Operating Characteristic curve to evaluate the model's ability to discriminate between classes.
- 
  ![krzywa_ROC](https://github.com/user-attachments/assets/ed0ecf88-f1b3-486b-872f-355dea4bac61)

