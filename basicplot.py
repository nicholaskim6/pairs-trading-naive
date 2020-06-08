import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('basics.csv')

n = 170

holdReturns = 0.94003
strat1Returns = 1.019
strat1Vol = 0.05071

strat2Returns = 1.033
strat1Vol = 0.0218


hold = [0.94003]
strat1 = [1.019]
strat2 = [1.033]

index = ['Returns']
df = pd.DataFrame({
                    'strat1' : strat1,
                    'strat2' : strat2,
                    'Hold' : hold}, index = index)
df.plot.bar(rot=0)
plt.show()
