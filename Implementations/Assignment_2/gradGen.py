import numpy as np
import pandas as pd
from matplotlib import pyplot
from math import sqrt
import itertools


def plotData(x, y):
    pyplot.plot(x, y)
    pyplot.ylabel('Y - axis')
    pyplot.xlabel('X - axis')
    pyplot.show()

def errorFunction(X, y, theta):
    m = y.shape[0]  # number of training examples
    hypothesis = np.dot(X, theta)
    cost = np.sum(np.square(hypothesis - y)) / (2 * m)
    return cost

def gradientDescent(X, y, theta, alpha, iters):
    m = y.shape[0]
    theta = theta.copy()
    error_history = []
    xtrans = np.transpose(X)
    print("theta is ", theta)
    for i in range(iters):
        hypothesis = np.dot(X, theta)
        loss = hypothesis - y
        gradient = np.dot(xtrans, loss) / m
        theta = theta - alpha * gradient
        err = errorFunction(X, y, theta)
        print("iter no is ", i)
        print("theta - ", theta, " error - ", err)
        error_history.append(err)

    return theta, error_history


def scaleFeatureAndGenerate(df, n):
    x1 = df['x1']
    x2 = df['x2']
    print(df.columns)
    df.drop(['x1', 'x2'], inplace=True, axis=1)
    count = 1
    for deg in range(1, n+1):
        for deg_x in range(0, deg + 1):
            count += 1
            deg_y = deg - deg_x
            str = f"x^{deg_x}" + f"y^{deg_y}"
            df[str] = (x1 ** deg_x) * (x2 ** deg_y)
            df[str] = (df[str] - np.mean(df[str])) / np.std(df[str])

    return df, count


def rmse(x, y, theta):
    theta = np.array(theta)
    hypothesis = np.dot(x, theta)
    loss = (y - hypothesis) ** 2
    ans = np.sum(loss) / 2 * y.shape[0]
    return sqrt(ans)


def main():
    df = pd.read_csv("./dataset/dataset.csv")
    n = int(input("enter the degree of polynomial"))
    m = len(df['B'])
    df.drop(['A'], inplace=True, axis=1)
    df.columns = ['x1', 'x2', 'y']
    df, count = scaleFeatureAndGenerate(df, n)
    df.insert(0, column='x0', value=np.ones(m))
    y = df['y']
    df.drop(['y'], inplace=True, axis=1)
    mask = np.random.rand(len(df)) < 0.7
    x_train, y_train = df[mask], y[mask]
    x_test, y_test = df[~mask], y[~mask]
    
    X = df
    print(x_train.head())
    print(df.head())

    theta = np.zeros(count)
    iters = 1000
    alpha = 0.0001
    theta, error_history = gradientDescent(x_train, y_train, theta, alpha, iters)
    
    print("final theta", theta)
    plotData(range(len(error_history)), error_history)

    error = errorFunction(x_test, y_test, theta)
    print(sqrt(error))


if __name__ == "__main__":
    main()
