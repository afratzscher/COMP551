import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
# from IPython.core.debugger import set_trace #for debugging
euclidean = lambda x1, x2: np.sqrt(np.sum((x1-x2)**2, axis=-1))
manhattan = lambda x1, x2: np.sum(np.abs(x1-x2), axis=-1)

class KNN:
	def __init__(self, K=1, dist_fn = euclidean):
		self.dist_fn = dist_fn
		self.K = K
		return 
	def fit(self, x, y):
		''' This stores training data -> is because is a lazy learner'''
		self.x = x
		self.y = y
		self.C = np.max(y) + 1
		return self
	def predict(self, x_test):
		''' This makes prediction using stored training data and the test
		data given as argument'''
		num_test = x_test.shape[0] # number of test points
		''' 
		calculate distance between training and test samples
		-> returns array of shape [num_test, num_train]
		NOTE: use self.x[None, :, :] and x_test[:, None, :] b/c different shapes
		'''
		distances = self.dist_fn(self.x[None, :, :], x_test[:, None, :]) 
			# NOTE: for each test point x', compare to every training point xi & record
		
		# initialize with 0s -> will have i-th row of knns stores indices of k closest training samples to i-th test sample
		knns = np.zeros((num_test, self.K), dtype=int)
		#initialize -> will have ith row of yprob has probability distribution over C classes
		y_prob = np.zeros((num_test, self.C))

		# fill
		for i in range(num_test):
			knns[i, :] = np.argsort(distances[i])[:self.K]
				# this gets indices of points (1st = closest, last = furthest)
				# NOTE: np.argsort(distances[i]) sorts by index (i.e. first = index of closest point)
				# NOTE: [:self.K] gets only k closest neighbours
				# repeat for all test points
			y_prob[i, :] = np.bincount(self.y[knns[i, :]], minlength=self.C)
				# Counts number of instances of each class in k-closest training samples
				# self.y[...] gets class of each of the k nearest neighbors (ie. [1 1 1])
				# then, counts number of class = 0, # class = 1, ...
				# stores in [a, b, c, ...] where a = # class 0, ...
			
		# divide yprob by K to get probability distribution
		y_prob /= self.K
		return y_prob, knns

def initialize():
	np.random.seed(1234) #use for reproducability 

def getdata():
	return datasets.load_iris()

def visual_initial(x_train, y_train, x_test, y_test):
	# c = ... means color based on class instead of train vs test
	plt.scatter(x_train[:,0], x_train[:, 1], c=y_train, marker='o', label='train')
	plt.scatter(x_test[:,0], x_test[:, 1], c=y_test, marker='s', label='test')
	plt.legend()
	plt.ylabel('sepal length')
	plt.xlabel('sepal width')
	plt.show()

def run(dataset):
	''' EXTRACT DATA'''
	# assumes only using sepal length and sepal width (for simpler visualization)
	# x = <sepal length, sepal width, petal length, petal width> -> found in DESCR
	x,y = dataset['data'][:, :2], dataset['target'] # data put into array
	(N,D), C = x.shape, np.max(y)+1 # counts instances, features (from X), classes (from Y)
	# print(f'instances (N) \t {N} \n features (D) \t {D} \n classes (C) \t {C}')
	
	''' SPLIT DATA'''
	inds = np.random.permutation(N) # generates indices arrayfrom 0 to N-1 and permutes it
	x_train, y_train = x[inds[:100]], y[inds[:100]]
	x_test, y_test = x[inds[100:]], y[inds[100:]]

	''' VISUALIZE DATA '''
	# visual_initial(x_train, y_train, x_test, y_test)

	''' RUN KNN'''
	model = KNN(K=3)
	y_prob, knns = model.fit(x_train, y_train).predict(x_test)
	print('knns shape:', knns.shape)
	print('y_prob shape', y_prob.shape)

	''' EVALUATE KNN'''
	y_pred = np.argmax(y_prob, axis=-1)
		# choose class with max probability
	accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
	print(f'accuracy is {accuracy*100:.1f}.')

	correct = y_test==y_pred
	incorrect = np.logical_not(correct) # stores data (NOT count)

	plt.scatter(x_train[:,0], x_train[:,1], c=y_train, marker='o', alpha=.2, label='train')
	plt.scatter(x_test[correct,0], x_test[correct,1], marker='.', c=y_pred[correct], label='correct')
	plt.scatter(x_test[incorrect,0], x_test[incorrect,1], marker='x', c=y_test[incorrect], label='misclassified')
		# keeps if index = True in correct/incorrect array

	# connect each node to k-nearest neighbours in training set
	for i in range(x_test.shape[0]):
		for k in range(model.K):
			hor = x_test[i, 0], x_train[knns[i,k],0]
			ver = x_test[i,1], x_train[knns[i,k],1]
			plt.plot(hor, ver, 'k-', alpha=.1)
	plt.ylabel('sepal length')
	plt.xlabel('sepal width')
	plt.legend()
	# plt.show()

	''' DECISION BOUNDARIES'''
	plt.clf()
	# create mesh grid -> if increase # samples to > 200, makes grid finer
		#mesh grid = like line paper but have dots instead of lines
	x0v = np.linspace(np.min(x[:, 0]), np.max(x[:,0]), 200)
	x1v = np.linspace(np.min(x[:, 1]), np.max(x[:,1]), 200)
		# generates linear sequence from min and max x values 
			#e.g. linspace(1,3,5) gives 1,1.5,2,2.5,3
	
	# to features values as a mesh
	x0, x1 = np.meshgrid(x0v, x1v)
	x_all = np.vstack((x0.ravel(), x1.ravel())).T
		# NOTE: .T transposes 
		# creates grid -> color each based on y_pred_all (found later)

	for k in range(1,4):
		model = KNN(K = k)
		y_train_prob = np.zeros((y_train.shape[0], C)) #initialize
		y_train_prob[np.arange(y_train.shape[0]), y_train] = 1
			# fills table given class of points in y_train (true values)
			# np.arange(...) gives indices of all points in ascending order

		# to get class probability of all points in 2D grid
		y_prob_all, _ = model.fit(x_train, y_train).predict(x_all)
			#predicts all data using model trained on subset
		y_pred_all = np.zeros_like(y_prob_all) #initialize
			#NOTE: zeros_like fills array with 0s
		y_pred_all[np.arange(x_all.shape[0]), np.argmax(y_prob_all, axis=-1)] = 1
			#for each data point (test and train), find max probability
			#and set corresponding cell (0, 1, or 2) to 1 for max prob class
		
		''' PLOTTING'''
		plt.scatter(x_train[:,0], x_train[:,1], c=y_train_prob, marker='o', alpha=1) # plots data
		plt.scatter(x_all[:,0], x_all[:,1], c=y_pred_all, marker='.', alpha=0.01) # shows boundary
		''' can also plot class probabilities (diff values of probabilities take diff colours)'''
		# plt.scatter(x_all[:,0], x_all[:,1], c=y_prob_all, marker='.', alpha=0.01)
		plt.ylabel('sepal length')
		plt.xlabel('sepal width')
		# plt.show()

	'''EFFECTS OF NOISE AND FEATURE SCALING'''
	plt.clf() # clear plot
	noise_scale = [0.01, 0.1, 1, 10, 100, 1000]
	noise = np.random.randn(x.shape[0], 1)
	results = []
		#generate random noise
	for s in noise_scale: # testing for different scale for noise
		x_n = np.column_stack((x, noise*s))
		results.append([]) # holds accuracy for different runs
		#repeats exp. 100x with diff train/test split
		for r in range(100):
			inds = np.random.permutation(N)
			x_train, y_train = x_n[inds[:100]], y[inds[:100]]
			x_test, y_test = x_n[inds[100:]], y[inds[100:]]
			#define model
			model = KNN(K=3)
			# prediction
			y_prob, _ = model.fit(x_train, y_train).predict(x_test)
			y_pred = np.argmax(y_prob, 1)
			accuracy = np.sum(y_pred == y_test)/y_test.shape[0]
			results[-1].append(accuracy)
	results = np.array(results)
	plt.errorbar(noise_scale, results.mean(1), results.std(1))
	plt.xscale('log')
	plt.xlabel("scale of noise frequency")
	plt.ylabel("accuracy")
	plt.show()
	#RESULT:
		#see that higher scale = more important role when calculating distance ->
		# so, want LOW scale for noisy features, HIGH scale for important ones 

def main():
	initialize()
	dataset = getdata()
	# print(dataset.DESCR) # this gives description of dataset
	# help(dataset) # help
	run(dataset)
	print("END")

if __name__ == "__main__":
	main()
