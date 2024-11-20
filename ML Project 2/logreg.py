import numpy as np

class LogisticRegression:

    def __init__(self, alpha=0.01, regLambda=0.01, epsilon=0.0001, maxNumIters=100):

        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
        self.JHist = []
        self.theta = None
        self.xMean = None
        self.xStd = None

    def computeCost(self, theta, X, y, regLambda):
 
        m = len(y)
        predictions = self.sigmoid(X @ theta)
        error = -y * np.log(predictions) - (1 - y) * np.log(1 - predictions)
        cost = error.mean() + (regLambda / (2 * m)) * np.sum(theta[1:]**2)
        return cost

    def computeGradient(self, theta, X, y, regLambda):

        m, d = X.shape
        predictions = self.sigmoid(X @ theta)
        error = X.T @ (predictions - y) / m
        reg_term = (regLambda / m) * np.concatenate([[0], theta[1:]])
        gradient = error + reg_term
        return gradient


    def fit(self, X, y):

        n, d = X.shape
        self.xMean = np.mean(X, axis=0)
        self.xStd = np.std(X, axis=0)
        X = (X - self.xMean) / self.xStd
        X = np.c_[np.ones((n, 1)), X]  
        self.theta = np.zeros(d + 1)  

        for i in range(self.maxNumIters):
            gradient = self.computeGradient(self.theta, X, y, self.regLambda)
            self.theta -= self.alpha * gradient
            cost = self.computeCost(self.theta, X, y, self.regLambda)
            self.JHist.append(cost)

            if np.linalg.norm(gradient) < self.epsilon:
                break

    def predict(self, X):
   
        n, d = X.shape
        X = (X - self.xMean) / self.xStd
        X = np.c_[np.ones((n, 1)), X]  
        predictions = self.sigmoid(X @ self.theta)
        return (predictions >= 0.5).astype(int)
    
    
    def sigmoid(self, z):

        return 1 / (1 + np.exp(-z))
