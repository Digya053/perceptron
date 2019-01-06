import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


class Perceptron:

    def __init__(self, data, labels, df, outfile='output1.csv'):
        self.data = data
        self.labels = labels
        self.weights = np.zeros(3)
        self.df = df
        self.outfile = outfile
        self.predicted_values = []

    def train(self):
        weights = []
        self.outfile = open(self.outfile, 'w+')
        while True:
            w_o = self.weights
            self.predicted_values = []
            for i in range(len(self.data)):
                value = np.dot(self.weights, self.data[i])
                prediction = 1 if value > 0 else -1
                self.predicted_values.append(value)
                if prediction * self.labels[i] <= 0:
                    self.weights = self.weights + self.labels[i] * self.data[i]

                self.visualize()
            deviation = np.linalg.norm(self.weights - w_o, ord=1)
            self.save_to_file()
            if deviation == 0:
                break

    def visualize(self):
        plt.scatter(self.df['x1'], self.df['x2'], c=self.df['y'])

        plt.xlabel('x1')
        plt.ylabel('x2')

        x1 = np.linspace(
            np.min(
                self.predicted_values), np.max(
                self.predicted_values), 100)
        x2 = (-self.weights[0] - self.weights[1] * x1) / self.weights[2]

        plt.plot(x1, x2)
        plt.show()

    def save_to_file(self):
        self.outfile.write(str(self.weights[1]) +
                           ',' +
                           str(self.weights[2]) +
                           ',' +
                           str(self.weights[0]) +
                           '\n')


if __name__ == "__main__":
    in_file = sys.argv[1]
    out_file = sys.argv[2]

    weights = []

    df = pd.read_csv(in_file, names=['x1', 'x2', 'y'])
    df.insert(0, 'bias', np.ones(df.shape[0]))

    data = df.iloc[:, 0:3].as_matrix()
    y = df['y'].values

    p = Perceptron(data, y, df)
    p.train()
