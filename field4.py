import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
from credit_card_data import read_data

pd.set_printoptions(max_rows=50, max_columns=21)

df = read_data()
pos = df[df.anomaly]
neg = df[df.anomaly == False]

pmf_pos = pos.groupby('field4').size().apply(lambda x: float(x) / len(pos))
pmf_neg = neg.groupby('field4').size().apply(lambda x: float(x) / len(neg))
pmf_neg.plot(c='b'); pmf_pos.plot(c='r')
plt.show()
plt.clf()

def discrete_distribution(xk, pk):
	return stats.rv_discrete(name='hour1',values=(xk,pk))

xk_pos = tuple(pmf_pos.index)
pk_pos = tuple(pmf_pos.values)
rv_pos = discrete_distribution(xk_pos, pk_pos)

xk_neg = tuple(pmf_neg.index)
pk_neg = tuple(pmf_neg.values)
rv_neg = discrete_distribution(xk_neg, pk_neg)

# estimate the mean value of `field4` for anomalous transactions
field4_means_pos = np.mean(rv_pos.rvs(size=len(pos)))

# estimate the mean value of `field4` for non-anomalous transactions
field4_means_neg = np.mean(rv_neg.rvs(size=len(neg)))

print 'Generating random variates from the same prob distribution'
print "Estimated difference in the means"
print field4_means_pos - field4_means_neg

print "Observed difference in the means"
print pos.field4.mean() - neg.field4.mean()
print

print "Pooling anomalies with non-anomalies"
print "generating samples of equal size to observed `neg` and `pos` sizes"
print "computing the means"

rows = np.array(random.sample(df.index, len(pos)))
prows = df.ix[rows]
nrows = df.drop(rows)

pmf_pos = prows.groupby('field4').size().apply(lambda x: float(x) / len(prows))
pmf_neg = nrows.groupby('field4').size().apply(lambda x: float(x) / len(nrows))
pmf_neg.plot(c='b'); pmf_pos.plot(c='r')
plt.show()
plt.clf()

xk_pos = tuple(pmf_pos.index)
pk_pos = tuple(pmf_pos.values)
rv_pos = discrete_distribution(xk_pos, pk_pos)

xk_neg = tuple(pmf_neg.index)
pk_neg = tuple(pmf_neg.values)
rv_neg = discrete_distribution(xk_neg, pk_neg)

# estimate the mean value of `field4` for anomalous transactions
field4_means_pos = np.mean(rv_pos.rvs(size=len(prows)))

# estimate the mean value of `field4` for non-anomalous transactions
field4_means_neg = np.mean(rv_neg.rvs(size=len(nrows)))

print "estimated difference in the means"
print field4_means_pos - field4_means_neg

print "observed difference in the means"
print pos.field4.mean() - neg.field4.mean()

# plt.hist(field4_means_pos)
# plt.hist(field4_means_neg)
