import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./data/DataminingContest2009.Task1.Train.Inputs", sep=",")
pd.set_printoptions(max_rows=25, max_columns=len(df.columns))

df.indicator1 = df.indicator1.astype(bool)
df.indicator2 = df.indicator2.astype(bool)

targets = pd.read_csv("./data/DataminingContest2009.Task1.Train.Targets", sep=",", columns=["anomaly"])
targets.anomaly = targets.anomaly.astype(bool)

amount_not_total = df[df.amount != df.total]
domains = df.domain1.unique()
domains = domains.head(20)

print "Number of Observations: %d" % len(df)
print "Number of Anomalies: %d" % targets[targets.anomaly].count()

# the histogram of the data
n, bins, patches = plt.hist(df.field4, 38, normed=1, facecolor='g', alpha=0.75)
plt.xlabel('field4', fontsize=14, color='blue')
plt.ylabel('Probability', fontsize=14, color='blue')
plt.title('Histogram of field4', fontsize=16, color='black')
plt.axis([0, 50, 0, 0.16])
plt.grid(True)

# bar plot of indicator1 and indicator2 fields
# which we know nothing about
selected = df[df.domain1.isin(domains.index)]
indicator1s = selected.groupby(['domain1'])['indicator1'].sum()
indicator2s = selected.groupby(['domain1'])['indicator2'].sum()
indicators = pd.DataFrame(indicator1s).join(pd.DataFrame(indicator2s))

# the histogram of the data
ind, width = np.arange(len(indicators)), 0.35
fig = plt.figure(figsize=(10,5))

ax = fig.add_subplot(111)
rects1 = ax.bar(ind, indicators.indicator1, width, color='r')
rects2 = ax.bar(ind+width, indicators.indicator2, width, color='y')
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(ind+width)
ax.set_xticklabels(indicators.index, rotation=45)