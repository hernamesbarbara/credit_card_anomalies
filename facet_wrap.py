import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import logit, ols
from patsy import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

df = pd.DataFrame(100 * np.random.rand(100, 4), index=range(100), columns=['A', 'B', 'C', 'D'])
df.A = 1000 * np.random.rand(100,)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot("121")
ax1.plot(df.A)

ax2 = fig.add_subplot("122", sharex=ax1, sharey=ax1)
ax2.plot(df.B)

formatter = ticker.FormatStrFormatter("$%1.2f")
plt.gca().yaxis.set_major_formatter(formatter)

plt.show()