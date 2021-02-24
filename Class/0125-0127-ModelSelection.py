import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, neighbors, tree, model_selection

# define function for MSE loss
loss = lambda y, yh: np.mean((y-yh)**2)

def initialize():
	np.random.seed(1234)

def mseK(x,y):
	(num_instances, num_features), num_classes = x.shape, np.max(y)+1
	'''SPLIT DATA'''
	n_test = num_instances//5 # 20% for testing, 80% for training
	inds = np.random.permutation(num_instances)
	x_train, y_train = x[inds[n_test:]], y[inds[n_test:]]
	x_test, y_test = x[inds[:n_test]], y[inds[:n_test]]

	K_list = range(1,100)
	err_train, err_test = [], []
	for i, K in enumerate(K_list):
		model = neighbors.KNeighborsRegressor(n_neighbors=K)
		model = model.fit(x_train, y_train)
		err_test.append(loss(model.predict(x_test), y_test))
		err_train.append(loss(model.predict(x_train), y_train))
	plt.plot(K_list, err_test, '-', label='unseen')
	plt.plot(K_list, err_train, '-', label='train')
	plt.legend()
	plt.xlabel('K (number of neighbours)')
	plt.ylabel('mean squared error')
	plt.show()

def validation(x,y):
	(num_instances, num_features), num_classes = x.shape, np.max(y)+1
	n_test, n_valid = num_instances//10, num_instances//10 
		# 80% train, 10% test, 10% valid
	inds = np.random.permutation(num_instances)
	x_test, y_test = x[inds[:n_test]], y[inds[:n_test]]
	x_valid, y_valid = x[inds[n_test:n_test+n_valid]], y[inds[n_test:n_test+n_valid]]
	x_train, y_train = x[inds[n_test+n_valid:]], y[inds[n_test+n_valid:]]

	K_list = range(1,30)
	err_train, err_test, err_valid = [], [], []
	for i, K in enumerate(K_list):
	    model = neighbors.KNeighborsRegressor(n_neighbors=K)
	    model = model.fit(x_train, y_train)
	    err_test.append(loss(model.predict(x_test), y_test))
	    err_valid.append(loss(model.predict(x_valid), y_valid))
	    err_train.append(loss(model.predict(x_train), y_train))
	    
	plt.plot(K_list, err_test,  label='test')
	#plt.plot(K_list, err_train,  label='train')
	plt.plot(K_list, err_valid, label='validation')

	plt.legend()
	plt.xlabel('K (number of neighbours)')
	plt.ylabel('mean squared error')
	plt.show()

def cross_validate(n, n_folds=10):
    n_val = n // n_folds
    inds = np.random.permutation(n)
    inds = []
    for f in range(n_folds):
        tr_inds = [] # train indices
        val_inds = list(range(f * n_val, (f+1)*n_val)) # validation indices
       
        if f > 0:
            tr_inds = list(range(f*n_val))
        if f < n_folds - 1:
            tr_inds = tr_inds + list(range((f+1)*n_val, n))
        yield tr_inds, val_inds
        	# yield is like return BUT lets function start where it left off

def cv(x,y):
	(num_instances, num_features), num_classes = x.shape, np.max(y)+1
	inds = np.random.permutation(num_instances)
	n_test, n_valid = num_instances // 10, num_instances // 10
		#8:1:1 split train:valid:test
	x_test, y_test = x[inds[:n_test]], y[inds[:n_test]]
	x_rest, y_rest = x[inds[n_test:]], y[inds[n_test:]]
	n_rest = num_instances - n_test #rest = valid + train

	num_folds = 10
	K_list = range(1,30)
	err_test, err_valid = np.zeros(len(K_list)), np.zeros((len(K_list), num_folds))
	for i, K in enumerate(K_list):
	    for f, (tr, val) in enumerate(cross_validate(n_rest, num_folds)):
	        model = neighbors.KNeighborsRegressor(n_neighbors=K)
	        model = model.fit(x_rest[tr], y_rest[tr])
	        err_valid[i, f] = loss(y_rest[val], model.predict(x_rest[val]))
	    model = neighbors.KNeighborsRegressor(n_neighbors=K)
	    model = model.fit(x_rest, y_rest)
	    err_test[i]= loss(y_test, model.predict(x_test))
	    
	plt.plot(K_list, err_test,  label='test')
	plt.errorbar(K_list, np.mean(err_valid, axis=1), np.std(err_valid, axis=1), label='validation')
	plt.legend()
	plt.xlabel('K (number of neighbours)')
	plt.ylabel('mean squared error')
	plt.show()

def confusion_matrix(y,yh):
	#rows = actual for each class
	#col = predicted for each class
	n_classes = np.max(y) + 1 # num classes
	c_matrix = np.zeros((n_classes, n_classes))
	for c1 in range(n_classes):
		for c2 in range(n_classes):
			#when both conditions are true, set to 1 (0 if not)
			c_matrix[c1,c2] = np.sum((y==c1)*(yh==c2))
	return c_matrix

def evaluation(x,y):
	x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)
		# splits 80% train, 20% test automatically
	yh_test = tree.DecisionTreeClassifier().fit(x_train, y_train).predict(x_test)
		#yh = yhat = predicted y
	cmat = confusion_matrix(y_test, yh_test)
	print(cmat)
	print(f'accuracy: {np.sum(np.diag(cmat))/np.sum(cmat)}')

def run():
	'''FETCH DATA'''
	x,y = datasets.fetch_california_housing(return_X_y=True)

	'''show train and test error NOT the same'''
	mseK(x,y)
	
	'''show validation is similar to test'''
	validation(x,y)

	'''Cross validation used to pick best model'''
	'''best model = simplest model with test error within 
	one stdev of model with lowest validation error'''
	cv(x,y)

	'''Evaluation'''
	x, y = datasets.load_iris(return_X_y=True)
	evaluation(x,y)

def main():
	initialize()
	run()

if __name__ == "__main__":
	main()