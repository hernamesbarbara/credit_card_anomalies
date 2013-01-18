import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import logit, ols


pd.set_printoptions(max_rows=25, max_columns=21)



def read_data():
	df = pd.read_csv("./data/DataminingContest2009.Task1.Train.Inputs", sep=",")
	targets = pd.read_csv("./data/DataminingContest2009.Task1.Train.Targets", sep=",", names=["anomaly"])
	df["anomaly"] = targets.anomaly
	bool_cols = [col for col in df.columns if np.all(df[col] < 2)]
	number_cols = ['amount', 'total', 'field3']
	
	for col in bool_cols:
		df[col] = df[col].astype(bool)
	
	cat_cols = [col for col in df.columns 
				if col not in number_cols and col not in bool_cols]
	return df, bool_cols, number_cols, cat_cols

df, bool_cols, number_cols, cat_cols = read_data()

print "Number of Observations: %d\n" % len(df)
print "Number of Anomalies: %d\n" % df.anomaly.sum()



def facet_by(frame, tobin="field5", facet="anomaly"):
	
	fig = plt.figure(figsize=(14, 7))
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
			ax  = fig.add_subplot(dims, sharex=ax, sharey=ax)
			ax.set_title(title_format %(subset[tobin].name, subset[facet].name, subset[facet].unique()[0]),
				fontsize=14, color='black')

		n, bins, patches = ax.hist(subset[tobin], normed=True, facecolor='g', alpha=0.75)
		ax.set_xlabel(subset[tobin].name, fontsize=12, color='blue')
		ax.set_ylabel('Probability', fontsize=12, color='blue')
		ax.grid(True)
	
	fig.suptitle("Distribution of %s" % subset[tobin].name, fontsize=16, color='black')
	plt.show()
	plt.clf()

	return

def plot_number_cols():
	[facet_by(df, tobin=col) for col in number_cols]
	return

print "Anomalies when field3 is Negative"
field3_negative =  df[df.field3 < 0 ]['anomaly']
grouped = field3_negative.value_counts()
print "%2d/100" % (100 * (float(grouped.values.min())/grouped.values.max()))
print

print "Anomalies when field3 is Positive"
field3_positive = df[df.field3 >= 0]['anomaly']
grouped = field3_positive.value_counts()
print "%2d/100" % (100 * (float(grouped.values.min())/grouped.values.max()))
print

# test if subset w/ negative `field3` as 
# significantly different distribution of anomalies
# if P <= 0.05, the distributions are significantly different
Z_statistic, P = stats.ranksums(x=field3_negative, y=field3_positive)
print "Tested if subset with `field3` < 0 has different distribution of anomalies"
print "\tdistributions are significantly different if P <= 0.05"
print "\tP => %f" % P






