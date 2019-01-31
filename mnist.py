#!/usr/bin/env python3

import numpy as np
import simpleNN.nn as nn
import simpleNN.Functions as Func


def predict(output):
    return output.argmax()


if __name__ == '__main__':
    inp_shape = np.array([32, 32])
    out_shape = np.array([10])
    NEpoch = 20

    # Prepare training data, training labels and test data, test labels
    training_data = np.zeros([10000, 32, 32])
    training_label = np.zeros([10000, 1])

    testing_data = np.zeros([5000, 32, 32])
    testing_label = np.zeros([5000, 1])

    layers = [32*32, 128, 32, 10]
    types = ['FC', 'FC', 'FC']
    activation_fn = ['sigmoid', 'sigmoid', 'sigmoid']
    learning_rate = 1e-3

    classifier = nn.NeuralNetwork(layers, types, activation_fn, learning_rate=learning_rate,
                                  predict_fn=predict, loss_fn=Func.CrossEntropy())

    print("Training starts...")
    for epoch in range(NEpoch):
        print("Epoch %d:" % epoch)
        classifier.train(training_data, training_label)

    print("Testing starts...")
    classifier.test(testing_data, testing_label)
