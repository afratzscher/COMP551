import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
# this is for BINARY classification (so uses)
# y-hat = sigma(W(sigma(Vx)))
# M hidden units
# assumes NO bias parameter for each layer

logistic = lambda z: 1./ (1 + np.exp(-z))
class MLP:
    def __init__(self, M=64): 
        self.M = M      
    def fit(self, x, y, optimizer):
        N,D = x.shape   
        #calculates gradient using backpropagation
        def gradient(x, y, params):
            v, w = params
            z = logistic(np.dot(x, v)) #N x M
            yh = logistic(np.dot(z, w))#N
            dy = yh - y #N
            dw = np.dot(z.T, dy)/N #M
            dz = np.outer(dy, w) #N x M
            dv = np.dot(x.T, dz * z * (1 - z))/N #D x M
            # dividing by N to average
            dparams = [dv, dw]
            return dparams
        #initially, have random weights 
        w = np.random.randn(self.M) * .01
        v = np.random.randn(D,self.M) * .01
        params0 = [v,w]
        # call gradient descent to get updated weights
        self.params = optimizer.run(gradient, x, y, params0)
        return self

    def predict(self,X):
        v,w = self.params
        z = logistic(np.dot(x,v))
        yh = logistic(np.dot(z,w))
        # b/c y-hat = logistic(W(logistic(Vx)))
        return yh
#define gradient descent
class GradientDescent:
    def __init__(self, learning_rate=.001, max_iters=1e4, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.epsilon = epsilon
    def run(self, gradient_fn, x, y, params):
        norms = np.array([np.inf])
        t = 1
        while np.any(norms > self.epsilon) and t < self.max_iters:
            grad = gradient_fn(x, y, params) #get gradient
            for p in range(len(params)):
                params[p] -= self.learning_rate * grad[p]
            t += 1
            norms = np.array([np.linalg.norm(g) for g in grad])
        return params
# to run
def main():
    dataset = datasets.load_iris()
    x, y = dataset['data'][:,[1,2]], dataset['target']
    y =  y == 1
    model = MLP(M=32)
    optimizer = GradientDescent(learning_rate=.1, max_iters=20000)
    yh = model.fit(x, y, optimizer).predict(x) 
    x0v = np.linspace(np.min(x[:,0]), np.max(x[:,0]), 200)
    x1v = np.linspace(np.min(x[:,1]), np.max(x[:,1]), 200)
    x0,x1 = np.meshgrid(x0v, x1v)
    x_all = np.vstack((x0.ravel(),x1.ravel())).T
    yh_all = model.predict(x_all) > .5
    plt.scatter(x[:,0], x[:,1], c=y, marker='o', alpha=1)
    plt.scatter(x_all[:,0], x_all[:,1], c=yh_all, marker='.', alpha=.01)
    plt.ylabel('sepal length')
    plt.xlabel('sepal width')
    plt.title('decision boundary of the MLP')
    plt.show()

if __name__ == "__main__":
    main()