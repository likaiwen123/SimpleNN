#!/usr/bin/env python3

import simpleNN.Layers as Layers
import numpy as np
import simpleNN.utils


class NeuralNetwork:
    def __init__(self, layers, types, activation_fns, learning_rate, predict_fn, loss_fn):
        self.number = len(layers) - 1
        if isinstance(types, str):
            types = [types]
            self.types = types * self.number
        elif len(types) == 1:
            types = list(types)
            self.types = types * self.number
        elif len(types) == self.number:
            self.types = list(types)
        else:
            raise ValueError

        if isinstance(activation_fns, str):
            activation_fns = [activation_fns]
            self.activation_functions = tuple(activation_fns * self.number)
        elif len(activation_fns) == 1:
            activation_fns = list(activation_fns)
            self.activation_functions = tuple(activation_fns * self.number)
        elif len(activation_fns) == self.number:
            self.activation_functions = list(activation_fns)
        else:
            raise ValueError

        if isinstance(learning_rate, float):
            learning_rate = [learning_rate]
            self.learning_rate = tuple(learning_rate * self.number)
        elif len(learning_rate) == 1:
            learning_rate = list(learning_rate)
            self.learning_rate = tuple(learning_rate * self.number)
        elif len(learning_rate) == self.number:
            self.learning_rate = list(learning_rate)
        else:
            raise ValueError

        self.shape = layers
        self.layers = [None] * self.number

        self.__build()

        self.predict_fn = predict_fn
        self.loss_fn = loss_fn

    def __build(self):
        for i in range(self.number):
            if self.types[i] == "FC":
                self.layers[i] = Layers.FCLayer(self.shape[i], self.shape[i + 1],
                                                activation_func=self.activation_functions[i],
                                                learning_rate=self.learning_rate[i])
            else:
                print("Layer type " + self.types[i] + " temporarily not supported.")
                raise TypeError

    def train(self, training_set, labels):
        correct = 0
        total_loss = 0.0
        for i in range(len(training_set)):
            image = training_set[i, :]

            self.clear_grad()

            output = image
            for l in range(self.number):
                output = self.layers[l].forward(output, grad=True)

            # prediction
            # output = simpleNN.utils.relu(output)
            output /= np.sum(output)
            predict = self.predict_fn(output)
            label = self.predict_fn(labels[i, :])
            if predict == label:
                correct += 1

            # loss calculation
            fact = labels[i, :]
            output = output.reshape(-1)
            loss = self.loss_fn.func(fact, output)
            diff = self.loss_fn.diff(fact, output)
            total_loss += loss

            # gradient decent and clear gradients
            prefix = diff.T
            for j in range(self.number):
                index = self.number - 1 - j
                prefix = self.layers[index].backward(prefix)
                self.layers[index].clear_grad()

        correct_rate = correct / len(training_set)
        average_loss = total_loss / len(training_set)

        print("Average loss is %10.6lf, correction rate %.2lf%%.\n" % (average_loss, correct_rate * 100))

    def test(self, testing_set, labels):
        correct = 0
        total_loss = 0.0
        predicts = np.zeros(len(testing_set))
        for i in range(len(testing_set)):
            image = testing_set[i, :]

            output = image
            for l in range(self.number):
                output = self.layers[l].forward(output, grad=False)

            # prediction
            # output = simpleNN.utils.relu(output) + 1e-10
            output /= np.sum(output)
            predicts[i] = self.predict_fn(output)
            label = self.predict_fn(labels[i, :])
            if predicts[i] == label:
                correct += 1

            # loss calculation
            fact = labels[i, :]
            output = output.reshape(-1)
            loss = self.loss_fn.func(fact, output)
            total_loss += loss

        correct_rate = correct / len(testing_set)
        average_loss = total_loss / len(testing_set)

        print("Average loss is %10.6lf, correction rate %.2lf%%.\n" % (average_loss, correct_rate * 100))

    def clear_grad(self):
        for i in range(self.number):
            self.layers[i].clear_grad()
