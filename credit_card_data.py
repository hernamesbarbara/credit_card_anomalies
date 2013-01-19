import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
pd.set_printoptions(max_rows=25, max_columns=21)

def read_data():
	f_train = "./data/DataminingContest2009.Task1.Train.Inputs"
	f_target = "./data/DataminingContest2009.Task1.Train.Targets"
	
	df = pd.read_csv(f_train, sep=",")
	targets = pd.read_csv(f_target, sep=",", names=["anomaly"])
	
	df["anomaly"] = targets.anomaly
	bool_cols = [col for col in df.columns if np.all(df[col] < 2)]
	number_cols = ['amount', 'total', 'field3']
	
	for col in bool_cols:
		df[col] = df[col].astype(bool)
	
	cat_cols = [ col for col in df.columns 
				 if col not in number_cols 
				 and col not in bool_cols ]

	return df

