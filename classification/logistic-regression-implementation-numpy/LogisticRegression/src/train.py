
import numpy as np
import os
os.listdir()
import src.dataset as dataset
import src.logisticregression as lr

data = dataset.SentimentDataset()
data.load_data()
data.clean_data()

logReg = lr.LogisticRegression()
trainX = logReg.vectorize(data.trainX)
J, theta = logReg.gradientDescent(trainX, data.trainY, np.zeros((trainX.shape[1], 1)), .01, 30)
print(f"The cost after training is {J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")

logReg.predict_sentiment(text=" Nice movie")
logReg.predict_sentiment("sad")
logReg.predict_sentiment("very bad")
logReg.predict_sentiment("very good")
logReg.predict_sentiment("nice")
logReg.predict_sentiment("good movie")

