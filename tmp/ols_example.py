import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import logit, ols
import matplotlib.pyplot as plt

df = pd.DataFrame(np.random.randn(100, 4), index=range(100), columns=['A', 'B', 'C', 'D'])

# two syntaxes for building 
# ordinary least squares models

y = df.A
x = df[["B", "C", "D"]]
model_a = sm.OLS(y,x).fit()
print model_a.summary() 
print

model_b = ols('y ~ B + C + D', df=df).fit()
print model_b.summary()
print

plt.plot(model_a.resid)
plt.show()

plt.plot(model_b.resid)
plt.show()
