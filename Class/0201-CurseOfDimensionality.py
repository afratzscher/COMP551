import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, neighbors, tree, model_selection

def run():
	'''FETCH DATA'''
	x,y = datasets.fetch_california_housing(return_X_y=True)


def main():
	initialize()
	run()

if __name__ == "__main__":
	main()