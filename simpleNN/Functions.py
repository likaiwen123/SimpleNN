#!/usr/bin/env python3

import simpleNN.utils as util


class Function:
    def __init__(self):
        self.func = None
        self.diff = None
        self.name = ''


class Sigmoid(Function):
    def __init__(self):
        super().__init__()
        self.func = util.sigmoid
        self.diff = util.sigmoid_diff
        self.name = 'Sigmoid'


class Tanh(Function):
    def __init__(self):
        super().__init__()
        self.func = util.tanh
        self.diff = util.tanh_diff
        self.name = 'Tanh'


class ReLU(Function):
    def __init__(self):
        super().__init__()
        self.func = util.relu
        self.diff = util.relu_diff
        self.name = 'ReLU'


class CrossEntropy(Function):
    def __init__(self):
        super().__init__()
        self.func = util.cross_entropy
        self.diff = util.cross_entropy_diff
        self.name = "Cross_Entropy"


class Self(Function):
    def __init__(self):
        super().__init__()
        self.func = util.self
        self.diff = util.self_diff
        self.name = "self"
