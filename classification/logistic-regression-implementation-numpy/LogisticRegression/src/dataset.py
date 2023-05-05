
import numpy as np
from string import punctuation
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import pandas as pd
from nltk.corpus import twitter_samples
import re
from nltk.corpus import stopwords


def clean_review(text):
    text = text.lower()
    # text = ''.join([c for c in text if c not in punctuation])
    ps = PorterStemmer()
    stopwords_english = stopwords.words('english')
    text = ''.join([c for c in text if c not in punctuation])
    # remove stock market tickers like $GE
    text = re.sub(r'\$\w*', '', text)
    text = re.sub(r'\@\w*', '', text)
    text = re.sub(r'\#\w*', '', text)
    # remove old style retweet text "RT"
    text = re.sub(r'^RT[\s]+', '', text)
    # remove hyperlinks
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    # remove hashtags
    # only removing the hash # sign from the word
    text = re.sub(r'#', '', text)
    text = ' '.join([ps.stem(x) for x in text.split() if x not in stopwords_english and x not in punctuation])
    text = ''.join([c for c in text if c not in punctuation])
    return text


class SentimentDataset:
    def __init__(self):
        self.trainX = None
        self.testX = None
        self.trainY = None
        self.testY = None

    def load_data(self):
        positive_tweets = twitter_samples.strings('positive_tweets.json')
        negative_tweets = twitter_samples.strings('negative_tweets.json')
        test_pos = positive_tweets[4000:]
        train_pos = positive_tweets[:4000]
        test_neg = negative_tweets[4000:]
        train_neg = negative_tweets[:4000]
        self.trainX = train_pos + train_neg
        self.testX = test_pos + test_neg
        self.trainY = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
        self.testY = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)


    def clean_data(self):
        self.trainX = [clean_review(x) for x in self.trainX]
        self.testX = [clean_review(x) for x in self.testX]



