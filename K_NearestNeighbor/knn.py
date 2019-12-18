# -*- coding: utf-8 -*-
# @Time    : 12/18/19 6:00 PM
# @Author  : zhongyuan
# @Email   : zhongyuandt@gmail.com
# @File    : knn.py
# @Software: PyCharm

import scipy.spatial.kdtree as kdtree
import os
import matplotlib.pyplot as plt


class KNN(object):
    def __init__(self,input,target):
        self.input = input
        self.target = target
        self.buildKDTree()

    def buildKDTree(self):
        print("building kdtree...")
        self.kdtree = kdtree.KDTree(self.input,leafsize=2048)

    def forecast(self,data,k=3):
        _,loc = self.kdtree.query(data,k)

        positive = 0
        nagtive = 0
        for i in loc:
            if self.target[i] == 1:
                positive+=1
            else:
                nagtive+=1
        if positive > nagtive:
            return True
        else:
            return False

    def draw(self,data):
        positive = []
        nagtive = []

        for i in range(len(data)):
            if self.forecast(data[i]):
                positive.append(data[i])
            else:
                nagtive.append(data[i])

        plt.figure(1)
        x1 = [x[0] for x in positive]
        x2 = [x[1] for x in positive]
        plt.scatter(x1, x2, label="positive", color="g", s=30, marker="o")
        x1 = [x[0] for x in nagtive]
        x2 = [x[1] for x in nagtive]
        plt.scatter(x1, x2, label="nagtive", color="r", s=30, marker="x")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.axis([value_range[0], value_range[1], value_range[0], value_range[1]])
        def f(x):
            return -x
        x = np.array(range(value_range[0],value_range[1]+1))
        plt.plot(x,f(x),"b-",lw=2)
        plt.title = "KNN"
        plt.legend()
        if not os.path.exists("model"):
            os.mkdir("model")
        plt.savefig("model/result.png")
        plt.show()

if __name__ == "__main__":
    import numpy as np

    value_range = [-10, 10]
    def f(x1, x2):
        if (x1 + x2) >= 0:
            return 1
        else:
            return -1
    input = np.random.uniform(value_range[0], value_range[1], (100, 2))
    target = [f(x1, x2) for x1, x2 in input]

    model = KNN(input,target)

    test_data = np.random.uniform(value_range[0], value_range[1], (100, 2))
    model.draw(test_data)