#!/usr/bin/env python

import numpy as np
import simpleNN.Functions as Func
import abc


class Layer:
    def __init__(self, inp_shape, out_shape, activation_func, learning_rate=1e-3):
        self.inp_shape = inp_shape
        self.out_shape = out_shape
        self.inp = np.zeros(self.inp_shape)
        self.out = np.zeros(self.out_shape)
        self.kernel = np.array([], dtype=np.double)
        self.grad = np.array([], dtype=np.double)
        self.learning_rate = learning_rate

        if activation_func.lower() == "sigmoid":
            self.activation = Func.Sigmoid()
        elif activation_func.lower() == "tanh":
            self.activation = Func.Tanh()
        elif activation_func.lower() == "relu":
            self.activation = Func.ReLU()
        else:
            print("Type of activation function not supported: " +
                  activation_func)
            raise ValueError

        self.type = "layer"

    def clear_grad(self):
        self.grad *= 0.0

    def forward(self, inp, grad=False):
        self.inp = inp
        self.out = self.activation.func(self.linear())
        if grad:
            self.calc_local_grad()
        return self.out

    def backward(self, vec):
        res = self.back(vec)
        self.update()
        return res

    @abc.abstractmethod
    def back(self, vec):
        """
        Implemented in method of subclasses.
        """

    def update(self):
        self.kernel += -self.grad * self.learning_rate

    @abc.abstractmethod
    def linear(self):
        """
        Implemented in method of subclasses.
        """

    @abc.abstractmethod
    def calc_local_grad(self):
        """
        Implemented in method of subclasses.
        """


class FCLayer(Layer):
    def __init__(self, inp_size, out_size, activation_func, learning_rate=1e-3):
        if type(inp_size) != int or type(out_size) != int:
            print("Input size parameters for FC Layer should be integers!")
            raise TypeError

        inp_shape = (inp_size,)
        out_shape = (out_size,)
        super().__init__(inp_shape, out_shape, activation_func, learning_rate)

        self.shape = (out_size, inp_size)
        self.kernel = np.random.randn(out_size, inp_size)
        self.__local_grad_coeff = np.zeros(self.shape)
        self.__local_grad_inp = np.zeros(self.shape)
        self.grad = np.zeros(self.shape)

        self.type = "FC"

    def calc_local_grad(self):
        diff = self.activation.diff(self.out)
        self.__local_grad_coeff = np.outer(diff, self.inp)
        self.__local_grad_inp = np.diag(diff).dot(self.kernel)

    def linear(self):
        self.inp = np.reshape(self.inp, (-1, 1))
        res = np.zeros(self.out_shape)
        for index in range(self.out_shape[0]):
            res[index] = np.sum(np.multiply(self.inp, self.kernel[:, index]))
        return res

    def back(self, vec):
        remote_grad = np.diag(vec)
        self.grad += remote_grad.dot(self.__local_grad_coeff)
        return vec.dot(self.__local_grad_inp)
