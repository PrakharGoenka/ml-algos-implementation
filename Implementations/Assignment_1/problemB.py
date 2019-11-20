import numpy as np
import pandas as pd
import math


def error(x, y, theta):
    m = y.shape[0]
    hypothesis = np.dot(x, theta)
    cost = np.sum(np.square(hypothesis - y)) / (2 * m)
    return cost


def scaleFeature(df):
    x1 = df['x1']
    x2 = df['x2']
    mean_x1, std_x1 = np.mean(x1), np.std(x1)
    mean_x2, std_x2 = np.mean(x2), np.std(x2)
    df['x1'] = (x1 - mean_x1) /std_x1
    df['x2'] = (x2 - mean_x2) / std_x2
    return df


def gradientDescent(x, y, theta, alpha, iters):
    m = y.shape[0]
    theta = theta.copy()
    error_history = []
    xtrans = np.transpose(x)

    for i in range(iters):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        print(loss)
        print(xtrans)
        # exit()    
        gradient = np.dot(np.transpose(x.sample()), loss) / m
        theta = theta - alpha * gradient
        err = error(x, y, theta)
        print("iter - ", i)
        print("theta - ", theta)
        error_history.append(err)

    return theta, error_history


def rmse(x, y, theta):
    theta = np.array(theta)
    hypothesis = np.dot(x, theta)
    loss = (y - hypothesis) ** 2
    ans = np.sum(loss) / y.shape[0]
    return math.sqrt(ans)


def main():
    df = pd.read_csv("./dataset/dataset.csv")
    x0 = np.ones(df.shape[0])

    df.drop(['A'], inplace=True, axis=1)
    df.columns = ['x1', 'x2', 'y']
    df.insert(0, column='x0', value=x0)
    target = df['y']
    df.drop(['y'], inplace=True, axis=1)

    mask = np.random.rand(len(df)) < 0.7

    x_train, y_train = df[mask], target[mask]
    x_test, y_test = df[~mask], target[~mask]

    theta = np.zeros(3)
    iters = 1000
    alpha = 0.00001

    theta, _ = gradientDescent(x_train, y_train, theta, alpha, iters)
    print("final theta ", theta)

    error = rmse(x_test, y_test, theta)
    print(error)


if __name__ == "__main__":
    main()
