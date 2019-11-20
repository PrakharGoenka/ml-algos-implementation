import numpy as np
import pandas as pd
from matplotlib import pyplot
from math import sqrt


def plotData(x, y):
    fig = pyplot.figure()
    pyplot.plot(x, y, 'ro', ms=10, mec='k')
    pyplot.ylabel('Y - axis')
    pyplot.xlabel('X - axis')

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

def scaleFeature(df):
    x1 = df['x1']
    x2 = df['x2']
    mean_x1, std_x1 = np.mean(x1), np.std(x1)
    mean_x2, std_x2 = np.mean(x2), np.std(x2)
    df['x1'] = (x1 - mean_x1) /std_x1
    df['x2'] = (x2 - mean_x2) / std_x2
    return df

def rmse(theta, x, y):
    theta = np.array(theta)
    hypothesis = np.dot(x, theta)
    loss = (y - hypothesis) ** 2
    ans = np.sum(loss) / y.shape[0]
    return sqrt(ans)

def main():
    df = pd.read_csv("./dataset/dataset.csv")

    m = len(df['B'])
    df.drop(['A'], inplace=True, axis=1)
    df.columns = ['x1', 'x2', 'y']
    df.insert(0, column='x0', value=np.ones(m))
    y = df['y']
    df.drop(['y'], inplace=True, axis=1)
    df = scaleFeature(df)
    mask = np.random.rand(len(df)) < 0.7
    x_train, y_train = df[mask], y[mask]
    x_test, y_test = df[~mask], y[~mask]
    
    X = df
    print(df.head())

    theta = np.zeros(3)
    iters = 1000
    alpha = 0.0001
    theta, error_history = gradientDescent(x_train, y_train, theta, alpha, iters)

    # theta = [2.1114, 0.0811, -0.1856 ]
    print("final theta {:.4f}, {:.4f}, {:.4f}".format(*theta))

    error = rmse(theta, x_test, y_test)
    print(error)


if __name__ == "__main__":
    main()
