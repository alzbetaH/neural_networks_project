# Neural Network Implementation in C++

This project implements a fully connected deep neural network in C++ to solve
the Fashion MNIST dataset.

After training for a user-defined number of epochs, evaluates the model on both
the training and testing datasets to output the predictions to
`train_predictions.csv` and `test_predictions.csv`.

With the current hyperparameters, we achieve 88.04% accuracy on the testing set
after around 2 minutes of CPU time.


# Requirements

### Fashion MNIST Dataset

Expected location:
```
data/fashion_mnist_test_labels.csv
data/fashion_mnist_test_vectors.csv
data/fashion_mnist_train_labels.csv
data/fashion_mnist_train_vectors.csv
```


# Execution

### Using a Makefile

Simply run `make run` to compile and run the neural network.

### Manually

Compile the source, eg. using `make`, to generate `network` executable.

Then, the usage is:
`./network -e [NUM_EPOCHS] -l [LEARNING_RATE] -b [BATCH_SIZE] INPUT_NEURONS_AMOUNT HIDDEN_LAYER_1_NEURONS_AMOUNT [...] OUTPUT_NEURONS_AMOUNT`


# Network Details

Validation set, 20% of the training set, is used to calculate the accuracy
after each batch (batching is implemented).

For weight initialization, He weight init is used. ReLU activation function is
used for hidden layers, and softmax for the output layer. The categorical cross
entropy was chosen for the loss function. The network uses SGD with momentum
and RMSProp. Dropout is implemented but turned off as it doesn't seem
necessary.


# Sources

Code architecture inspired by
[https://www.youtube.com/watch?v=sK9AbJ4P8ao](https://www.youtube.com/watch?v=sK9AbJ4P8ao)
