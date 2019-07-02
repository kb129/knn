from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import sys

class kNN:
    _dists = []
    def __init__(self, train_img, train_l):
        self._train_img = train_img
        self._train_l = train_l
    
    def _dist(self, p, q):
        # Euclidean distance
        return np.linalg.norm(p-q, ord=2)

    def predict(self, k, x, i):
        # make dists table
        if len(self._dists) < i+1:
            self._dists.append(np.array([self._dist(np.ravel(j), np.ravel(x)) for j in self._train_img]))

        # sort
        nearest_indexes = self._dists[i].argsort()[:k]
        nearest_labels = self._train_l[nearest_indexes]
        
        c = Counter(nearest_labels)
        # return most common label
        return c.most_common(1)[0][0]


def main():
    # load data
    (train_img, train_l), (test_img, test_l) = mnist.load_data()

    # show data num, label num
    print("train img num  :", train_img.shape)
    print("train label num:", train_l.shape)
    print("test img num   :", test_img.shape)
    print("test label num :", test_l.shape)

    # make instance
    model = kNN(train_img, train_l)

    # learning process
    for k in range(1, test_img.shape[0]):
        error_count = 0
        # create model
        print("{} NN start".format(k))
        for i in range(0, test_img.shape[0]):
            label = model.predict(k, test_img[i], i)
            if test_l[i] == label:
                error_count += error_count + 1
            sys.stdout.write('\r'+'{}%        '.format(100 * i/test_img.shape[0]))

        print("")
        # show img complete parsent
        print("error rate:{}".format(error_count / test_img.shape[0]))

if __name__ == '__main__':
    main()
