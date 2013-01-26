import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from credit_card_data import read_data

pd.set_printoptions(max_rows=50, max_columns=21)

def make_hist(frame, to_bin, norm=False):
	
	fig = plt.figure(figsize=(10, 4))
	ax = fig.add_subplot("111")
	
	ylab = 'Probability' if norm else 'Count'
	xlab = frame[to_bin].name

	ax.set_ylabel(ylab, fontsize=12, color='blue')
	ax.set_xlabel(xlab, fontsize=12, color='blue')

	ticks = range(24) # military times
	labels = map(lambda x: str("%s:00" %x), ticks)
	plt.xticks(ticks, labels)

	if not norm: ax.set_ylim(0, 8000)

	for tick in plt.gca().xaxis.iter_ticks():
	    tick[0].label2On = True
	    tick[0].label1On = False
	    tick[0].label2.set_rotation('vertical')

	n, bins, patches = ax.hist(frame[to_bin],
		bins=24,normed=norm, facecolor='g', alpha=0.75)

	ax.grid(True)
	plt.show()
	plt.clf()


df = read_data()
pos = df[df.anomaly]
neg = df[df.anomaly == False]


hour1 = df.groupby(['anomaly', 'hour1']).size().reset_index(name='freq')
most_anoms = hour1.groupby(['anomaly'])['freq'].idxmax()

print
print "Most Common Hour for Regular Payment Behavior: %d:00" % most_anoms[False]
print "Most Common Hour for Anomalous Behavior: %d:00" % most_anoms[True]
print
df.groupby(['anomaly']).apply(make_hist, "hour1", norm=True)

print
print "Most Common Hour for Regular Payment Behavior: %d:00" % most_anoms[False]
print "Most Common Hour for Anomalous Behavior: %d:00" % most_anoms[True]
print
df.groupby(['anomaly']).apply(make_hist, to_bin="hour2", norm=True)

# make pmfs for hour of the day
# for both anomalies and normal credit card transactions
pmf_pos = pos.groupby('hour1').size().apply(lambda x: float(x) / len(pos))
pmf_neg = neg.groupby('hour1').size().apply(lambda x: float(x) / len(neg))

pmf_neg.plot(c='b'); pmf_pos.plot(c='r')

xk = tuple(pmf_pos.index)
pk = tuple(pmf_pos.values)
rv = stats.rv_discrete(name='hour1',values=(xk,pk))





