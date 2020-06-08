import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import pandas as pd

import os

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing

from statsmodels.tsa.stattools import coint

from scipy import stats

temp = 0
prices = []
dirString = r'C:\Users\Nick Kim\Desktop\Kaggle Data\StocksNew'
directory = os.path.expanduser(r'C:\Users\Nick Kim\Desktop\Kaggle Data\StocksNew')
i = 0
names = []
for filename in os.listdir(directory):
    names.append(filename)
    temp = pd.read_csv(directory + '\\' + filename)
    prices.append(temp['Close'].tolist())
    i += 1
    if (i > 100):
        break

indices = []
for i, list in enumerate(prices):
    if (len(list) < 100):
        indices.append(i)

prices = [d for (i, d) in enumerate(prices) if i not in indices]
names = [d for (i, d) in enumerate(names) if i not in indices]


for i, list in enumerate(prices):
    prices[i] = prices[i][len(prices[i])-100:len(prices[i])]

# This gives us 100 last prices of this group of stocks
returns = []
for list in prices:
    returnAdd = []
    for i, price in enumerate(list):
        if (i == 0):
            continue
        else:
            returnAdd.append((list[i]-list[i-1])/list[i-1])
    returns.append(returnAdd)


#find cointegrations for 0
indices = []
coints = []
for i in range(len(returns)):
    val = coint(returns[0], returns[i])[1]
    thing = []
    thing.append(i)
    thing.append(val)
    coints.append(thing)

print(coints)
plt.plot(returns[0])
plt.plot(returns[80])
plt.show()
