#!/usr/bin/env python3

import numpy as np
import simpleNN.nn as nn
import simpleNN.Functions as Func


def predict(output):
    return output.argmax()


if __name__ == '__main__':
    inp_shape = [28, 28]
    out_shape = [10]
    NEpoch = 50

    # Prepare training data, training labels and test data, test labels
    training_data = np.load('data/training_images.npy').astype(np.double) / 256.0
    train_label = np.load('data/training_labels.npy')

    testing_data = np.load('data/testing_images.npy').astype(np.double) / 256.0
    test_label = np.load('data/testing_labels.npy')

    training_label = np.zeros((len(train_label), out_shape[0]), dtype=np.double)
    testing_label = np.zeros((len(test_label), out_shape[0]), dtype=np.double)
    for i in range(len(train_label)):
        training_label[i, train_label[i]] = 1.0
    for i in range(len(test_label)):
        testing_label[i, test_label[i]] = 1.0

    # training_data = training_data[:10, :]
    # training_label = training_label[:10, :]
    # testing_data = testing_data[:5, :]
    # testing_label = testing_label[:5, :]
    #
    layers = [28 * 28, 10]
    types = ['FC']
    activation_fn = ['sigmoid']
    learning_rate = 1e-2

    # layers = [28*28, 128, 32, 10]
    # types = ['FC', 'FC', 'FC']
    # activation_fn = ['sigmoid', 'sigmoid', 'sigmoid']
    # learning_rate = 1e-3

    classifier = nn.NeuralNetwork(layers, types, activation_fn, learning_rate=learning_rate,
                                  predict_fn=predict, loss_fn=Func.CrossEntropy())

    print("Training starts...")
    for epoch in range(NEpoch):
        print("Epoch %d:" % (epoch + 1))
        classifier.train(training_data, training_label)

    print("Testing starts...")
    classifier.test(testing_data, testing_label)
