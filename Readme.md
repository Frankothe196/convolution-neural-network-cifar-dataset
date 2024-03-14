# CIFAR-10 Image Classification with Convolution Neural Networks

This project demonstrates image classification on the CIFAR-10 dataset using a Convolutional Neural Network (CNN) built with TensorFlow Keras.

## Model Architecture

The CNN architecture used in this project is inspired by the VGG architecture. It consists of the following layers:

* Input layer with shape (32, 32, 3) for RGB images.
* Two sets of convolutional layers followed by batch normalization and MaxPooling2D layers.
* A flatten layer to convert the 2D output from the convolutional layers to a 1D vector.
* A dropout layer with a rate to prevent overfitting.
* A dense layer with 1024 units and a ReLU activation function.
* A final dense layer with 10 units and a softmax activation function for classifying the image into one of the 10 CIFAR-10 classes.

## Training

The model is trained on the CIFAR-10 training set for 10 epochs using the Adam optimizer and sparse categorical crossentropy loss. The validation set is used to monitor the model's performance during training.

## Data Augmentation

Data augmentation is applied to the training set to improve the model's generalization ability. This involves randomly shifting and flipping the images.

## Results

The model achieves an accuracy of around 75% on the CIFAR-10 test set. The confusion matrix shows that the model performs well on most classes, but it has some difficulty distinguishing between certain classes such as cats and dogs. This is expected as the Cifar dataset is of low quality making some images industinguishable.
