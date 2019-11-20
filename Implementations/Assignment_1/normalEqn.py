import numpy as np
import pandas as pd
from math import sqrt


def normalEqn(x, y):
    theta = np.zeros(x.shape[1])
    xtrans = np.transpose(x)
    temp = np.dot((xtrans), x)
    inv_temp = np.linalg.pinv(temp)
    temp = np.dot(inv_temp, xtrans)
    theta = np.dot(temp, y)
    return theta


def rmse(x, y, theta):
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
    mask = np.random.rand(len(df)) < 0.7
    x_train, y_train = df[mask], y[mask]
    x_test, y_test = df[~mask], y[~mask]

    theta = normalEqn(x_train, y_train)
    print(theta)
    print(rmse(x_test, y_test, theta))



if __name__ == "__main__":
    main()
