import numpy as np
import TextClassification.NaiveBayes.src.dataset as dataset

class NaiveBayes:
    def __init__(self):
        self.freqs = {}
        self.logprior = None
        self.loglikelihood = None
        self.vocab = None

    def create_freqs(self, texts, ys):
        for y, text in zip(ys, texts):
            for word in dataset.clean_review(text).split():
                pair = (word, y[0])
                if pair in self.freqs:
                    self.freqs[pair] += 1
                else:
                    self.freqs[pair] = 1

    def train(self, train_x, train_y):
        '''
        Input:
            freqs: dictionary from (word, label) to how often the word appears
            train_x: a list of tweets
            train_y: a list of labels correponding to the tweets (0,1)
        Output:
            logprior: the log prior. (equation 3 above)
            loglikelihood: the log likelihood of you Naive bayes equation. (equation 6 above)
        '''
        self.loglikelihood = {}
        self.logprior = 0

        # calculate V, the number of unique words in the vocabulary
        self.vocab = set([pair[0] for pair in self.freqs.keys()])
        V = len(self.vocab)

        # calculate N_pos and N_neg
        N_pos = N_neg = 0
        for pair in self.freqs.keys():
            # if the label is positive (greater than zero)
            if pair[1] > 0:
                # Increment the number of positive words by the count for this (word, label) pair
                N_pos += float(self.freqs.get(pair))

            # else, the label is negative
            else:
                # increment the number of negative words by the count for this (word,label) pair
                N_neg += float(self.freqs.get(pair))

        # Calculate D, the number of documents
        D = len(train_x)
        # Calculate D_pos, the number of positive documents
        D_pos = np.sum(np.array(train_y))

        # Calculate D_neg, the number of negative documents
        D_neg = D - D_pos

        # Calculate logprior
        self.logprior = np.log(D_pos) - np.log(D_neg)

        # For each word in the vocabulary...
        for word in self.vocab:
            # get the positive and negative frequency of the word
            freq_pos = self.freqs.get((word, 1), 0)
            freq_neg = self.freqs.get((word, 0), 0)

            # calculate the probability that each word is positive, and negative
            p_w_pos = (freq_pos + 1) / (N_pos + V)
            p_w_neg = (freq_neg + 1) / (N_neg + V)

            # calculate the log likelihood of the word
            self.loglikelihood[word] = np.log(p_w_pos) - np.log(p_w_neg)


    def predict(self, text):
        '''
        Input:
            tweet: a string
            logprior: a number
            loglikelihood: a dictionary of words mapping to numbers
        Output:
            p: the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)

        '''

        word_l = dataset.clean_review(text)
        # initialize probability to zero
        p = 0
        # add the logprior
        p += self.logprior
        for word in word_l.split():
            # check if the word exists in the loglikelihood dictionary
            if word in self.loglikelihood:
                # add the log likelihood of that word to the probability
                p += self.loglikelihood.get(word, 0)

        if p > 0:
            return "Review is positive"
        else:
            return "Review is Negative"