import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
from scipy.stats import norm
from matplotlib.legend_handler import HandlerLine2D


class Bayes(object):
    def __init__(self):
        self.clf = GaussianNB()

    def histo_plot(self, data):
        (mu, sigma) = norm.fit(data)
        # the histogram of the data
        n, bins, patches = plt.hist(data, 30, normed=1, facecolor='green', alpha=0.75)

        # add a 'best fit' line
        y = mlab.normpdf( bins, mu, sigma)
        l = plt.plot(bins, y, 'r--', linewidth=2)

    def training(self):
        # Training data: a and b
        # b is twice, three times, four times of a
        a = np.random.normal(1, 0.1, 3000)
        b = np.append(np.random.normal(2, 0.2, 1000), np.random.normal(3, 0.3, 1000))
        b = np.append(b, np.random.normal(4, 0.4, 1000))

        # X: b/a, Y: type, 2/3/4
        X = b/a
        Y = np.append(np.ones(1000) * 2, np.ones(1000) * 3)
        Y = np.append(Y, np.ones(1000) * 4)

        # Plot the histogram of b/a
        self.histo_plot(X[:1000])
        self.histo_plot(X[1000:2000])
        self.histo_plot(X[2000:3000])
        plt.xlabel('b/a')
        plt.ylabel('Probability')
        plt.title('Histogram of the training set b/a')
        plt.grid(True)
        plt.show()

        # Plot b/a
        plt.plot(X)
        plt.plot(Y, marker='o', markersize=5, label='Type')
        plt.legend()
        plt.title("Training Set b/a, and types")
        plt.ylabel("b/a")
        plt.xlabel("time (s)")
        plt.show()

        # Bayes Classifier
        X = X.reshape(-1, 1)
        self.clf.partial_fit(X, Y, np.unique(Y))
        print self.clf.class_prior_
        print self.clf.theta_
        print self.clf.sigma_

    def testing(self):
        # Testing data, c and d
        c = np.random.normal(2, 0.2, 300)
        d = np.append(np.random.normal(4, 0.4, 100), np.random.normal(6, 1, 100))
        d = np.append(d, np.random.normal(8, 1, 100))

        # X1: testing set, Y1: predicted result
        X1 = d/c
        Y1 = []
        for item in X1:
            Y1.append(self.clf.predict([[item]]))

        # Plot the histogram of d/c
        self.histo_plot(X1[:100])
        self.histo_plot(X1[100:200])
        self.histo_plot(X1[200:300])
        plt.xlabel('d/c')
        plt.ylabel('Probability')
        plt.title('Histogram of the testing set d/c')
        plt.grid(True)
        plt.show()

        # Plot d/c
        plt.plot(X1)
        plt.plot(Y1, marker='o', markersize=5, label='Type')
        plt.legend()
        plt.title("Testing Set d/c, and types")
        plt.ylabel("d/c")
        plt.xlabel("time (s)")
        plt.show()

def main():
    bayes = Bayes()
    bayes.training()
    bayes.testing()

if __name__ == "__main__":
    main()
