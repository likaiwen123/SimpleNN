#!/usr/bin/env python3

import numpy as np


# Sigmoid Function and its differentiate
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_diff(y):
    return y * (1.0 - y)


# tanh Function and its differentiate
def tanh(x):
    return np.tanh(x)


def tanh_diff(y):
    return 1.0 - np.power(y, 2)


# ReLU Function and its differentiate
def relu(x):
    signs = (x >= 0)
    return signs * x


def relu_diff(y):
    return y >= 0


# Self Function and its differentiate
def self(x):
    return x


def self_diff(y):
    return np.ones(y.shape)


# Cross Entropy Function and its differentiate
def cross_entropy(x_fact, x_pred):
    return -x_fact.dot(np.log(x_pred)) - (1.0 - x_fact).dot(np.log(1.0 - x_pred))


def cross_entropy_diff(x_fact, x_pred):
    return -np.divide(x_fact, x_pred) - np.divide(1.0 - x_fact, -(1.0 - x_pred))
