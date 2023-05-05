# vectorizer = CountVectorizer()
# vectorizer = TfidfVectorizer()
# vectorizer = vectorizer.fit(data.trainX)
# trainX = vectorizer.transform(data.trainX)
# trainX = trainX.toarray()

def predict_sentiment(text = " happy"):

    s = dataset.clean_review(text)
    x = vectorizer.transform([s]).toarray()
    z = np.dot(x, theta)
    h = lr.sigmoid(z)
    if h >=0.5:
        print("Positive")
    else:
        print("Negative")