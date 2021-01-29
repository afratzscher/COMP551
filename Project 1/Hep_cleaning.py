import numpy as np
import pandas as pd
import math
import seaborn as sns
import statistics
from scipy import stats
from sklearn.preprocessing import StandardScaler
from pandas.plotting import parallel_coordinates

import matplotlib.pyplot as plt

np.random.seed(123)
def removeMissing():
	df2 = pd.read_csv('hepatitis.csv')
	print(df2.shape[0], 'initial rows')
	df2 = df2.replace('?', np.nan)
	df2 = df2.dropna()
	df2.reset_index(drop=True, inplace=True) #reset indices (so first instance at index 0, next at index 1...)
	df2 = df2.astype(float) # all values are float (instead of string)
	print(df2.shape[0], ' rows after remove none')
	return df2	

def visualization(df2):
	print(df2)
	class1 = df2[df2['Class' ] == 1]
	class2 = df2[df2['Class'] == 2]
	stat1 = round(class1.describe(),2)
	stat2 = round(class2.describe(),2)
	total = pd.concat([stat1, stat2], axis=1, keys=['Class 1 Stats', 'Class 2 Stats'])

	ss = StandardScaler()
	cols = ['AGE', 'BILIRUBIN', 'ALK_PHOSPHATE', 'SGOT', 'ALBUMIN', 'PROTIME']
	subsetdf = df2[cols]
	scaled_df = ss.fit_transform(subsetdf)
	scaled_df = pd.DataFrame(scaled_df, columns=cols)
	plotdf = pd.concat([scaled_df, df2['Class']], axis=1)
	plotdf.head()
	pc = parallel_coordinates(df2, 'Class', color=('#FFE888', '#FF9999'))
	plt.show()

def outliers(df2):
	df2 = df2[(np.abs(stats.zscore(df2))<3).all(axis=1)]
	print(df2.shape[0], ' rows after outlier removal')
	
	corr_matrix = df2.corr()
	plt.clf()
	sns.heatmap(corr_matrix)
	plt.show()
	return df2

def main():
	df2 = removeMissing()
	visualization(df2)
	df2 = outliers(df2)

if __name__ == "__main__":
	main()