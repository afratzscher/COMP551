import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

class Node:
	def __init__(self, data_indices, parent):
		#stores data indices which are in region defined by node
		self.data_indices = data_indices
		# stores left child of node
		self.left = None
		# stores right child of node
		self.right = None
		# stores feature for split at this node
		self.split_feature = None
		# stores value of feature for split at this node
		self.split_value = None

		if parent:
			self.depth = parent.depth + 1
			self.num_classes = parent.num_classes
			self.data = parent.data
			self.labels = parent.labels
			class_prob = np.bincount(self.labels[data_indices], minlength=self.num_classes)
				#counts freq of different labels in region defined by node
			self.class_prob = class_prob / np.sum(class_prob)
				# NOTE: use class prob of elaf nodes to make predictions

class DecisionTree:
	def __init__(self, num_classes=None, max_depth=3, cost_fn=cost_misclassification, min_leaf_instances=1):
		self.max_depth = max_depth


def initialize():
	np.random.seed(1234)

def greedy_test(node, costfn):
	bestcost = np.inf
	bestfeature, bestvalue = None, None
	numinstances, numfeatures = node.data.shape
	data_sorted = np.sort(node.data[node.data_indices], axis=0)
	testcandidates = (data_sorted[1:] + data_sorted[:-1]) / 2.

def run():
	''' FETCH DATA'''
	dataset = datasets.load_iris()
	x,y = dataset['data'][:, :2], dataset['target']
	
	'''SPLIT DATA'''
	(num_instances, num_features), num_classes = x.shape, np.max(y)+1
	inds = np.random.permutation(num_instances)
	x_train, y_train = x[inds[:100]], y[inds[:100]]
	x_test, y_test = x[inds[100:]], y[inds[100:]]

	'''DEFINE DECISION TREE'''	
	tree = DecisionTree(max_depth=20)
	print('here')

def main():
	initialize()
	run()

if __name__ == "__main__":
	main()
