import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
import matplotlib.ticker as mt


def beta_distribution(a, b, x):
    '''
        Function to calculate Beta(x| a,b)
    '''
    val = (gamma(a+b) * (x ** (a-1)) * ((1-x) ** (b-1))) / (gamma(a) * gamma(b))
    return val


def gen_dataset(N, u_min, u_max):
    '''
        Funcion to return a randomly generated dataset for N coin tosses
        having m heads and l = N - m tails, where μML ∉ (u_min, u_max)
    '''
    # randomly select μML(u) according to given constraints
    u = np.random.random()
    while(u >= u_min and u <= u_max):
        u = np.random.random()

    m = int(N * u)

    # acertain that rounding off m, does not shift u in the prohibited range
    if m == N * u_min:
        m -= 1
    if m == N * u_max:
        m += 1

    # initialize dataset with all tails
    dataset = [0] * N

    # randomly select the position of the m heads
    for i in np.random.randint(0, N, m):
        dataset[i] = 1

    # return dataset and number of heads
    return dataset, m
    

def plot(y, filename):
    '''
        function to plot y vs x
    '''
    # generate points for x axis
    x = []
    for i in range(101):
        x.append(i/100)

    plt.axis([0, 1, 0, 20])
    plt.plot(x, y)
    plt.savefig(filename)
    plt.close()


def sequential(dataset, N, m):
    '''
        Function to estimate posterior distribution using sequential learning.
    '''
    # hyperparameters for prior Beta distribution of μ
    a = 2
    b = 3

    # list of lists to store values of pdf for μ in range(0, 1) for N iterations
    result = []

    # list to store values of pdf for μ in range(0, 1) for one iteration
    temp = []

    # initial posterior = prior distribution
    for i in range(101):
        u = i/100
        temp.append(beta_distribution(a, b, u))

    result.append(temp)

    for n in dataset:
        # update posterior to be the prior distribution for next iteration
        a += n
        b += 1 - n
        temp = []

        for i in range(101):
            u = i/100
            temp.append(beta_distribution(a, b, u))

        result.append(temp)

    return result


def combined(dataset, N, m):
    '''
        Function to estimate posterior distribution by taking the whole data together.
    '''
    # hyperparameters for prior Beta distribution of μ
    a = 2
    b = 3

    # udpate hyperparameters according to the given dataset
    a += m
    b += N - m

    # list to store values of pdf for μ in range(0, 1)
    result = []

    # calculate value of pdf for μ in range(0, 1)
    for i in range(101):
        u = i/100
        result.append(beta_distribution(a, b, u))

    return result


def main():
    dataset, m = gen_dataset(160, 0.4, 0.6)

    # estimate distribution using sequential method
    print('For sequential\n')
    result_s = sequential(dataset, 160, m)
    print(result_s)

    # estimate distribution using combined method
    print('\n\nFor combined\n')
    result_c = combined(dataset, 160, m)
    print(result_c)


    # plot results for sequential method
    for i in range(0, 161, 20):
        plot(result_s[i], 'sequential_' + str(i) + '.png')

    plot(result_c, 'combined.png')


if __name__ == '__main__':
    main()
