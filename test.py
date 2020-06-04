import itertools
import scipy
import scipy.stats as stats
from scipy.stats import boltzmann
from scipy.stats import geom
import numpy as np
import matplotlib.pyplot as plt

lower, upper = 1, 5
mu, sigma = 3, 5
lower, upper = 1, 20
mu, sigma = 8, 5
X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)


fig, ax = plt.subplots(1, 1)
lambda_, N = 0.05, 60
bolt = np.arange(boltzmann.ppf(0.01, lambda_, N, 1), boltzmann.ppf(0.99, lambda_, N, 1))
ax.plot(bolt, boltzmann.pmf(bolt, lambda_, N, 1), 'bo', ms=8, label='boltzmann pmf')


plt.savefig("test.png")