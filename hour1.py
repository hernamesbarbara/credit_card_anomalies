import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from credit_card_data import read_data

pd.set_printoptions(max_rows=50, max_columns=21)

def make_hist(frame, norm=False):
	fig = plt.figure(figsize=(10, 4))
	ax = fig.add_subplot("111")

	ticks = range(24) # military times
	labels = map(lambda x: str("%s:00" %x), ticks)
	
	plt.xticks(ticks, labels)

	for tick in plt.gca().xaxis.iter_ticks():
	    tick[0].label2On = True
	    tick[0].label1On = False
	    tick[0].label2.set_rotation('vertical')

	if not norm: ax.set_ylim(0, 8000)

	n, bins, patches = ax.hist(frame["hour1"], bins=24,
		normed=norm, facecolor='g', alpha=0.75)

	ax.grid(True)
	plt.show()
	plt.clf()


df = read_data()
pos = df[df.anomaly]
neg = df[df.anomaly == False]


hours = df.groupby(['anomaly', 'hour1']).size().reset_index(name='freq')
max_anoms = hours.groupby(['anomaly'])['freq'].idxmax()

make_hist(neg, True)
make_hist(pos, True)

print "Most Common Hour for Regular Payment Behavior: %d:00" % max_anoms[False]
print "Most Common Hour for Anomalous Behavior: %d:00" % max_anoms[True]
