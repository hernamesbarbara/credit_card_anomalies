import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from credit_card_data import read_data

pd.set_printoptions(max_rows=25, max_columns=21)

df = read_data()

print "Number of Observations: %d\n" % len(df)
print "Number of Anomalies: %d\n" % df.anomaly.sum()

def facet_by(frame, tobin, facet):
	"""make a histogram of a column with optional\nsubsetting on a boolean"""
	fig = plt.figure(figsize=(14, 7))
	dims = 121
	title_format = 'hist of %s where `%s` is %s'

	subset_names = frame[facet].unique()
	subsets = [ frame[frame[facet] == subset] for subset in subset_names]
	
	for i, subset in enumerate(subsets):
		dims += i
		
		if i == 0:
			ax = fig.add_subplot("121")
			
			ax.set_title(title_format %(subset[tobin].name, 
				subset[facet].name, subset[facet].unique()[0]),fontsize=14, color='black')
		else:
			ax  = fig.add_subplot(dims, sharex=ax, sharey=ax)
			
			ax.set_title(title_format %(subset[tobin].name,
				subset[facet].name, subset[facet].unique()[0]),fontsize=14, color='black')

		n, bins, patches = ax.hist(subset[tobin], normed=True, facecolor='g', alpha=0.75)
		ax.set_xlabel(subset[tobin].name, fontsize=12, color='blue')
		ax.set_ylabel('Probability', fontsize=12, color='blue')
		ax.grid(True)
	
	fig.suptitle("Distribution of %s" % subset[tobin].name, fontsize=16, color='black')
	plt.show()
	plt.clf()

def call(frame, func, *args):
	func(frame, *args)

plot_combos = [ (col, "anomaly") for col in df.columns
				if df[col].dtype not in (str, object) and col != 'anomaly' ]

[ call(df, facet_by, *plot) for plot in plot_combos ]


def make_scatter(frame, x_y, colour=None):
	"""scatters two number columns on a dataframe"""	
	x, y = x_y

	fig = plt.figure(figsize=(10, 5))
	ax = fig.add_subplot("111")

	if colour is not None:
		colours = frame[colour].unique() 
		frame = frame.groupby(colour).reset_index()
		
		ax.set_color_cycle([cm(1.*i/len(colours)) for i in range(len(colours))])
		
		for i, colour in enumerate(colours):
			rgb = cm(1.* i /len(colours))
			X = frame.ix[(colour)][x]
			Y = frame.ix[(colour)][y]

			ax.scatter(X, Y, marker='o', c=rgb, s=20, alpha=0.75)
		
		ax.legend(colours)

	else:
		pass
	
	ax.set_xlabel(x.title())
	ax.set_ylabel(y.title())
	plt.show()
	plt.clf()

plot_combos = [ (make_scatter,df,( ("field3", "amount"), "anomaly")), ]

for func, frame, args in plot_combos:
    call(frame, func, *args)



