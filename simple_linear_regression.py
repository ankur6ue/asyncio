#!/usr/bin/env python3

# Client. Sends stuff from STDIN to the server.

import asyncio
import websockets

import dill
import numpy as np
import matplotlib.pyplot as plt
import random

def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    return (b_0, b_1)


def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color="m",
                marker="o", s=30)

    # predicted response vector
    y_pred = b[0] + b[1] * x

    # plotting the regression line
    plt.plot(x, y_pred, color="g")

    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')

    # function to show plot
    plt.show()



def main():
    # observations
    x = np.arange(0, 10, 0.1)
    m = 2
    b = 0.5
    y = m*x + b + np.random.normal(0,1,100)

    # two sets of random indices
    ind = np.random.permutation(np.arange(0,100))
    ind1 = ind[0:-1:2]
    ind2 = ind[1:-1:2]
    # estimating coefficients
    x1 = x[ind1]
    y1 = y[ind1]
    x2 = x[ind2]
    y2 = y[ind2]
    b1 = estimate_coef(x1, y1)
    print("Estimated coefficients:\nb_0 = {}  \
    \nb_1 = {}".format(b[0], b[1]))

    # plotting regression line
    plot_regression_line(x, y, b)
    print('done')

if __name__ == "__main__":
    main()