import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import logit, ols

df = pd.read_csv("./data/DataminingContest2009.Task1.Train.Inputs", sep=",")
pd.set_printoptions(max_rows=25, max_columns=21)

targets = pd.read_csv("./data/DataminingContest2009.Task1.Train.Targets", sep=",", names=["anomaly"])
targets.anomaly = targets.anomaly.astype(bool)

df["anomaly"] = targets.anomaly

df.indicator1 = df.indicator1.astype(bool)
df.indicator2 = df.indicator2.astype(bool)

bool_cols = [col for col in df.columns if np.all(df[col] < 2)]
print
print "Boolean Columns: %s" % str(bool_cols)
print

number_cols = ['amount', 'total', 'field3']
print "Numerical Columns: %s" % str(number_cols)
print

for col in bool_cols:
    df[col] = df[col].astype(bool)

cat_cols = [col for col in df.columns if col not in number_cols and col not in bool_cols]
print "Categorical Columns: %s" % str(cat_cols)
print
print
print "Number of Observations: %d" % len(df)
print "Number of Anomalies: %d" % targets[targets.anomaly].count()

def field4_hist():
	"histogram of field4"
	fig = plt.figure(figsize=(10, 5))
	ax = fig.add_subplot(1, 1, 1)

	binwidth = 1
	x_low = df.field4.min()
	x_high = df.field4.max() + binwidth

	n, bins, patches = ax.hist(df.field4, 
		bins=np.arange(x_low,x_high,binwidth), 
		normed=True, facecolor='g', alpha=0.75)

	ax.set_xlabel('field4', fontsize=14, color='blue')
	ax.set_ylabel('Probability', fontsize=14, color='blue')
	ax.set_title('Histogram of field4', fontsize=16, color='black')
	ax.grid(True)
	plt.show()
	return

def field5_hist():
	# histogram field5
	fig = plt.figure(figsize=(10, 5))
	ax = fig.add_subplot("111")

	n, bins, patches = ax.hist(df.field5, bins=np.arange(0,16,1), 
		normed=True, facecolor='g', alpha=0.75)

	ax.set_xlabel('field5', fontsize=14, color='blue')
	ax.set_ylabel('Probability', fontsize=14, color='blue')
	ax.set_title('Histogram of field5', fontsize=16, color='black')
	ax.grid(True)
	plt.show()
	return

# looking at individual domains
domains = df.groupby(["domain1"]).size()
domains = domains.order(ascending=False)
domains = domains.head(20)

# bar plot of indicator1 and indicator2 fields
# which we know nothing about
selected = df[df.domain1.isin(domains.index)]
indicator1s = selected.groupby(['domain1'])['indicator1'].sum()
indicator2s = selected.groupby(['domain1'])['indicator2'].sum()
indicators = pd.DataFrame(indicator1s).join(pd.DataFrame(indicator2s))

def indicators_by_domain(indicators=indicators):
	# the histogram of the data
	ind, width = np.arange(len(indicators)), 0.35
	fig = plt.figure(figsize=(10,5))

	ax = fig.add_subplot(1, 1, 1)
	rects1 = ax.bar(ind, indicators.indicator1, width, color='r')
	rects2 = ax.bar(ind+width, indicators.indicator2, width, color='y')
	ax.set_ylabel('Sum of Indicator Flags', fontsize=14, color='blue')
	ax.set_title('indicator1 and indicator2 flags by domain')
	ax.set_xticks(ind+width)
	ax.set_xticklabels(indicators.index, rotation=45, fontsize=10, color='blue')
	plt.show()
	return

def facet_by(frame, tobin="field5", facet="anomaly"):
	
	fig = plt.figure(figsize=(20, 10))
	dims = 121
	title_format = 'hist of %s where `%s` is %s'

	subset_names = frame[facet].unique()
	subsets = [ frame[frame[facet] == subset] for subset in subset_names]
	
	for i, subset in enumerate(subsets):
		dims += i
		
		if i == 0:
			ax = fig.add_subplot("121")
			ax.set_title(title_format %(subset[tobin].name, subset[facet].name, subset[facet].unique()[0]),
				fontsize=14, color='black')
		else:
			ax  = fig.add_subplot(str(dims), sharex=ax, sharey=ax)
			ax.set_title(title_format %(subset[tobin].name, subset[facet].name, subset[facet].unique()[0]),
				fontsize=14, color='black')

		n, bins, patches = ax.hist(subset[tobin], normed=True, facecolor='g', alpha=0.75)
		
		ax.set_xlabel(subset[tobin].name, fontsize=12, color='blue')
		ax.set_ylabel('Probability', fontsize=12, color='blue')
		ax.grid(True)
	
	fig.suptitle("Distribution of %s" % subset[tobin].name, fontsize=16, color='black')

	return


print "When amount != total...."
print df[df.amount != df.total].groupby("anomaly").size()
print

for col in number_cols:
    facet_by(df, tobin=col)

print "Anomalies when field3 is less than 10,000"
print df[df.field3 < -10000].groupby("anomaly").size()
print 100 * (119./1000)
print
print "Anomalies when field3 is greater than 10,000"
print df[df.field3 > -10000].groupby("anomaly").size()
print 100 * (1975./91587)

ols_model = ols('amount ~  total', df=df).fit()
print ols_model.summary()
print
plt.scatter(x=df.total, y=ols_model.fittedvalues)



