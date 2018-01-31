from Models import Models
import numpy as np
import tensorflow as tf


class CNN(Models):
    #Constructor

    def __init__(self, nodes , activationFunction= "ReLU"):
        """
        :param
        1. nodes: nodes list of Deep neural network.
        2. activationFunction : choice of activation function , default is ReLU
        """
        super().__init__()
        self.nodes = nodes
        self.activationFunction = activationFunction
        self.X = None
        self.Y = None

    def get_name(self):
        return "Convolution neural network"

    def fit(self, x, y):
        """
        :param
        1. x is input of train data set
        2. y is result of train data set
        """
        np.random.seed(777)

        self.X = x
        self.Y = y

    def predict(self, x):
        self.X = x

