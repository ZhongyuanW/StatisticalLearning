# -*- coding: utf-8 -*-
# @Time    : 12/18/19 3:44 PM
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : perceptron.py
# @Software: PyCharm

import numpy as np
import os
import matplotlib.pyplot as plt

class Perceptron(object):
    def __init__(self, input, target, lr=0.1, load=False):
        self.input = np.array(input)
        self.target = np.array(target)
        self.dimension = len(input[0])
        self.lr = np.array(lr)
        self.parameters = {"w":0,"b":0}
        self.load = load
        self.init_parameter()

    def init_parameter(self):
        # init parameters
        self.parameters["w"] = np.zeros(self.dimension)
        self.parameters["b"] = 0
        if self.load:
            self.parameters["w"] = np.load("model/w.npy")
            self.parameters["b"] = np.load("model/b.npy")

    def sgd_train(self):
        train_iter = 0
        while True:
            if not self.haveMisclassification():
                self.parameters["w"] /= self.parameters["w"][0]
                self.parameters["b"] /= self.parameters["w"][0]
                if not os.path.exists("model"):
                    os.mkdir("model")
                np.save("model/w", self.parameters["w"])
                np.save("model/b", self.parameters["b"])
                break
            for i in range(len(self.target)):
                train_iter += 1
                print("the %dth train iter"%(train_iter))
                inner = np.inner(self.parameters["w"],self.input[i])

                # the misclassification point
                if self.target[i] * (inner + self.parameters["b"]) <= 0:
                    self.parameters["w"] += self.lr * self.target[i] * self.input[i]
                    self.parameters["b"] += self.lr * self.target[i]

    def haveMisclassification(self):
        for i in range(len(self.target)):
            if self.target[i] * (np.inner(self.parameters["w"], self.input[i]) + self.parameters["b"]) <= 0:
                return True
        return False


    def draw(self):
        positive_input = []
        nagtive_input = []

        for i in range(len(self.target)):
            if self.target[i] == 1:
                positive_input.append(self.input[i])
            else:
                nagtive_input.append(self.input[i])

        plt.figure(1)
        x1 = [x[0] for x in positive_input]
        x2 = [x[1] for x in positive_input]
        plt.scatter(x1, x2, label="positive", color="g", s=30, marker="o")
        x1 = [x[0] for x in nagtive_input]
        x2 = [x[1] for x in nagtive_input]
        plt.scatter(x1, x2, label="nagtive", color="r", s=30, marker="x")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.axis([value_range[0],value_range[1],value_range[0],value_range[1]])
        def f(x):
            return -(self.parameters["b"] + self.parameters["w"][0]*x/self.parameters["w"][1])
        x = np.array(range(value_range[0],value_range[1]+1))
        plt.plot(x,f(x),"b-",lw=2)
        plt.title = "Perceptron"
        plt.legend()
        if not os.path.exists("model"):
            os.mkdir("model")
        plt.savefig("model/result.png")
        plt.show()

if __name__ == "__main__":

    value_range = [-10,10]
    def f(x1,x2):
        if (x1 + x2) >= 0:
            return 1
        else:
            return -1
    input = np.random.uniform(value_range[0],value_range[1],(100,2))

    target = [f(x1,x2) for x1,x2 in input]
    model = Perceptron(input,target,0.1)
    model.sgd_train()
    print("the learned parameters are:",model.parameters)
    model.draw()


