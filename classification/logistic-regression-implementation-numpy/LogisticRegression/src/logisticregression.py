import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import TextClassification.LogisticRegression.src.dataset as dataset

class LogisticRegression:

    def __init__(self):
        self.h = None
        self.vectorizer = None
        self.theta = None

    def sigmoid(self,z):
        # calculate the sigmoid of z
        h = 1 / ((1 + np.exp(-z)))
        return h

    def vectorize(self,X):
        self.vectorizer = TfidfVectorizer()
        self.vectorizer = self.vectorizer.fit(X)
        X = self.vectorizer.transform(X)
        X = X.toarray()
        return X

    def gradientDescent(self,x, y, theta, alpha, num_iters):
        m = x.shape[0]
        self.theta = theta
        for i in range(0, num_iters):
            z = np.dot(x, self.theta)
            h = self.sigmoid(z)
            # calculate the cost function
            J = (-1 / m) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))
            print(J)
            # update the weights theta
            self.theta = self.theta - ((alpha / m) * (np.dot(x.T, h - y)))
        J = float(J)
        return J, self.theta

    def predict_sentiment(self,text=" happy"):
        s = dataset.clean_review(text)
        x = self.vectorizer.transform([s]).toarray()
        z = np.dot(x, self.theta)
        # print(self.theta)
        h = self.sigmoid(z)
        if h >= 0.5:
            print("Positive")
        else:
            print("Negative")

# def sigmoid(z):
#     # calculate the sigmoid of z
#     h = 1 / ((1 + np.exp(-z)))
#     return h
#
# def gradientDescent(x, y, theta, alpha, num_iters):
#     m = x.shape[0]
#     for i in range(0, num_iters):
#         z = np.dot(x, theta)
#         h = sigmoid(z)
#         # calculate the cost function
#         J = (-1 / m) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))
#         print(J)
#         # update the weights theta
#         theta = theta - ((alpha / m) * (np.dot(x.T, h - y)))
#     J = float(J)
#     return J, theta
