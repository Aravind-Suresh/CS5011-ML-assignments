import sys,os
import numpy as np
import sklearn
import csv
from util import *

"""
Program to train a neural network on DS2 and report the performance
"""

class FullyConnectedLayer:

    def __init__(self, dim, acti_func, acti_func_prime):
        assert len(dim) == 2, "All are fully connected layers. Therefore dimensionality is %d" %(2)
        self.count = dim[1]
        self.b = np.random.random(dim[1])
        self.W = np.random.random((dim[1], dim[0]))
        self.f = acti_func
        self.fd = acti_func_prime

    def feed_forward(self, x):
        # print self.W.shape, x.shape, self.b.shape
        z = np.dot(self.W, x) + self.b
        a = self.f(z)
        self.x = x
        self.a = a
        self.z = z
        return z, a

    def back_prop(self, delta):
        # TODO: Remove hardcoding
        return np.dot(self.W.T, delta)*(self.x)*(1-self.x)

    def update(self, delta, alpha):
        del_w = np.dot(self.a.T, delta)
        del_b = delta

        self.b = self.b - (alpha*del_b)
        self.W = self.W - (alpha*del_w)

class Transformer:

    def __init__(self, x):
        x_min = x.min()
        x_max = x.max()
        m = -0.5*(x_max + x_min)
        s = 2./(x_max - x_min)

        self.params = { 'm': m, 's': s }

    def transform_input(self, x):
        return self.params['s']*(x+self.params['m'])

def softmax(v):
    s = np.sum(np.exp(v))
    return np.array([ np.exp(vv) for vv in v ])/s

class Config:
    nn_input_dim = 96  # input layer dimensionality
    nn_output_dim = 4  # output layer dimensionality
    epsilon = 1e-1
    reg_lambda = 0

class Network:

    def __init__(self, nn_hdim):
        # self.n_layers = len(sizes) # 96, 20, 4

        W1 = np.random.randn(Config.nn_input_dim, nn_hdim) / np.sqrt(Config.nn_input_dim)
        # W1 = 2*np.random.random((Config.nn_input_dim, nn_hdim))-1
        b1 = np.random.random((1, nn_hdim))
        W2 = np.random.randn(nn_hdim, Config.nn_output_dim) / np.sqrt(nn_hdim)
        # W2 = 2*np.random.random((nn_hdim, Config.nn_output_dim))
        b2 = np.random.random((1, Config.nn_output_dim))

        self.model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2 }

        # print self.model

        self.A = Collections.sigmoid
        self.Ad = Collections.sigmoid_prime

    def feed_forward(self, x):
        a = z = x
        A = []
        Z = []
        for W, b in zip(self.weights, self.biases):
            z = a.dot(W) + b
            Z.append(z)
            a = self.A(z)
            A.append(a)
        a = softmax(z)
        A[-1] = a
        return Z, A

    def predict(self, X):
        W1 = self.model['W1']
        b1 = self.model['b1']
        W2 = self.model['W2']
        b2 = self.model['b2']

        z1 = X.dot(W1) + b1
        a1 = Collections.sigmoid(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        labels = np.argmax(probs, axis=1)
        return labels

    def cost(self, X, Y):
        P = self.predict(X)
        d = P - Y
        cost = 0.5*np.linalg.norm(d)
        return cost

    def accuracy(self, X, Y):
        P = self.predict(X)
        return (100.*np.sum(P == Y)/Y.shape[0])

    def gradient_descent(self, X, Y, n_iter=10000):

        W1 = self.model['W1']
        b1 = self.model['b1']
        W2 = self.model['W2']
        b2 = self.model['b2']

        for i in range(n_iter):
            z1 = X.dot(W1) + b1
            a1 = Collections.sigmoid(z1)
            z2 = a1.dot(W2) + b2
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            n_samples = len(X)

            delta3 = probs
            # Accounting for softmax layer
            delta3[range(n_samples), Y] -= 1
            # Backpropagation
            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(W2.T) * a1 * (1-a1)
            dW1 = np.dot(X.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += Config.reg_lambda * W2
            dW1 += Config.reg_lambda * W1

            # Gradient descent parameter update
            W1 += -Config.epsilon * dW1
            b1 += -Config.epsilon * db1
            W2 += -Config.epsilon * dW2
            b2 += -Config.epsilon * db2

            self.model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2 }

            if not i % 3000 and i:
                Config.epsilon *= 0.5
                print 'Alpha changed:', Config.epsilon

            if not i % 100:
                print 'Iter:', i, 'Cost:', self.cost(X, Y), 'Acc:', self.accuracy(test_X, test_Y_labels)

class NeuralNetwork:

    def __init__(self, config, alpha, acti_func, acti_func_prime, cost_func, cost_func_prime):
        self.n_layers = len(config)
        self.config = config
        self.layers = []
        self.A = acti_func
        self.Ad = acti_func_prime
        self.C = cost_func
        self.Cd = cost_func_prime
        self.alpha = alpha

        for i in range(1, self.n_layers, 1):
            layer = FullyConnectedLayer((self.config[i-1], self.config[i]), self.A, self.Ad)
            self.layers.append(layer)

    def feed_forward(self, x):
        a = x
        for layer in self.layers:
            z, a = layer.feed_forward(a)

        return z, a

    def compute_delta(self, x, y):
        z_o, a_o = self.feed_forward(x)
        delta_o = self.Cd(a_o, y)*self.Ad(z_o)
        # print delta_o
        return delta_o, z_o

    def back_prop(self, delta, z_o):
        _delta = delta
        deltas = [_delta]
        for i in range(self.n_layers-2, -1, -1):
            layer = self.layers[i]
            # layer.update(_delta, self.alpha)
            _delta = layer.back_prop(_delta)
            deltas.append(_delta)
        return deltas

    def measure_cost(self, X, Y):
        costs = []
        for i in range(len(X)):
            x = X[i]
            y = Y[i]
            p = self.feed_forward(x)
            c = self.C(p, y)
            costs.append(c)
        return np.mean(costs)

    def train(self, _X, Y):
        self.transformer = Transformer(_X)
        X = self.transformer.transform_input(_X)
        # print _X
        # print X
        deltas_all = []
        for i in range(len(X)):
            x = X[i]
            y = Y[i]
            delta_o, z_o = self.compute_delta(x, y)
            # print len(delta_o)s
            deltas_all.append(self.back_prop(delta_o, z_o))
        deltas_all = np.array(deltas_all)
        deltas_bp = np.sum(deltas_all, axis=0)
        for i in range(self.n_layers-2, -1, -1):
            layer = self.layers[i]
            layer.update(deltas_bp[self.n_layers-2-i], self.alpha)
        # if not i % 100:
        #     print 'Cost %d :' % i, self.measure_cost(X, Y)
        print 'Cost :', self.measure_cost(X, Y)

    def predict_label(self, _x):
        x = self.transformer.transform_input(_x)
        _, p = self.feed_forward(x)
        label = np.argmax(p)
        return label

    def predict(self, _x):
        x = self.transformer.transform_input(_x)
        _, p = self.feed_forward(x)
        return p

class Collections:

    @staticmethod
    def sigmoid(x):
        return 1./(1+np.exp(-x))

    @staticmethod
    def sigmoid_prime(x):
        s = Collections.sigmoid(x)
        return s*(1-s)

    @staticmethod
    def cost(p, a):
        # p - predicted
        # a - actual
        d = np.array(p - a)
        return 0.5*np.dot(d.T, d)

    @staticmethod
    def cost_prime(p, a):
        # p - predicted
        # a - actual
        d = np.array(p - a)
        return d

# def read_data(fin_data_name, fin_labels_name):
#     """
#     A utility function to read data and identify the threshold for the classifier given the file paths to the train and test data
#     """
#     lines1 = open(fin_data_name, 'r').readlines()
#     lines2 = open(fin_labels_name, 'r').readlines()
#     X, Y = [], []
#     for l in lines1:
#         X.append(map(float, l.split(',')))
#     Y = map(lambda x: float(x.strip()), lines2)
#     thresh = np.mean(list(set(Y)))
#     Y = np.array(Y).reshape((-1, 1))
#     return X, Y, thresh


# data_path = sys.argv[1]
# train_X, train_Y, thresh = read_data(data_path + 'train.csv', data_path + 'train_labels.csv')
#
# train_X = np.array(train_X)
# train_Y = np.zeros(len(train_Y))

train_data_path = sys.argv[1]
test_data_path = sys.argv[2]

train_data = read_data(train_data_path)
test_data = read_data(test_data_path)
np.random.shuffle(train_data)
n_classes = len(np.unique(train_data[:, -1]))

train_X = np.array(train_data[:, :-1])
# transformer = Transformer(train_X)
# train_X = transformer.transform_input(train_X)

train_Y_labels = np.uint8(train_data[:, -1])
train_Y = np.array(map(lambda x: one_hot(int(x), n_classes), train_Y_labels))

test_X = test_data[:, :-1]
test_Y_labels = np.uint8(test_data[:, -1])
test_Y =  np.array(map(lambda x: one_hot(int(x), n_classes), test_data[:, -1]))

net = Network(20)
model = net.gradient_descent(train_X, train_Y_labels, n_iter=10000)

# config = [96, 20, 4]
# alpha = 1
# neural_network = NeuralNetwork(config, alpha, Collections.sigmoid, Collections.sigmoid_prime, Collections.cost, Collections.cost_prime)
#
# for i in range(1000):
#     neural_network.train(train_X, train_Y)

# y = train_Y
# p = np.array(np.array(map(neural_network.predict, train_X))>0.5, dtype=np.uint8).ravel()

# Predict

# train_Y_labels_pred = np.array(map(neural_network.predict_label, train_X))
# p = train_Y_labels_pred
# y = train_Y_labels
# print 'Accuracy: %.2f' % (100.*np.sum(p == y)/y.shape[0]), '%'

# if __name__ == '__main__':
#     main()
