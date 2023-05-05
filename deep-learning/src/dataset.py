
import numpy as np
from string import punctuation
from collections import Counter
import config

class SentimentDataset:
    def __init__(self):
        self.review = None
        self.labels = None
        self.reviews_split = None
        self.vocab_to_int = None
        self.int_to_vocab = None
        self.review_ints = None
        self.encoded_labels = None
        self.features = None

    def load_data(self):
        with open('./input/reviews.txt', 'r') as f:
            self.reviews = f.read()
        with open('./input/labels.txt', 'r') as f:
            self.labels = f.read()

    def clean_data(self):
        self.reviews = self.reviews.lower()
        self.all_text = ''.join([c for c in self.reviews if c not in punctuation])
        self.reviews_split = self.all_text.split('\n')
        self.all_text = ' '.join(self.reviews_split)
        self.words = self.all_text.split()
        # return self.reviews_split

    def vocab_dict(self):
        counts = Counter(self.words)
        vocab = sorted(counts, key=counts.get, reverse=True)
        self.vocab_to_int = {word: ii for ii, word in enumerate(vocab)}
        self.int_to_vocab = {ii: word for word,ii in self.vocab_to_int.items() }
        # return self.vocab_to_int, self.int_to_vocab

    def encode_text(self):
        self.review_ints = []
        for reviews in self.reviews_split:
            self.review_ints.append([self.vocab_to_int[word] for word in reviews.split()])
        # return self.review_ints

    def encode_label(self):
        self.labels_split = self.labels.split('\n')
        self.encoded_labels = np.array([1 if label == 'positive' else 0 for label in self.labels_split])
        # return self.encoded_labels

    def remove_outliers(self):
        review_lens = Counter([len(x) for x in self.review_ints])
        print("Zero-length reviews: {}".format(review_lens[0]))
        print("Maximum review length: {}".format(max(review_lens)))
        print('Number of reviews before removing outliers: ', len(self.review_ints))
        non_zero_idx = [ii for ii, review in enumerate(self.review_ints) if len(review) != 0]
        self.review_ints = [self.review_ints[ii] for ii in non_zero_idx]
        self.encoded_labels = np.array([self.encoded_labels[ii] for ii in non_zero_idx])
        print('Number of reviews after removing outliers: ', len(self.review_ints))

        # return self.review_ints, self.encoded_labels

    def pad_features(self, seq_length):
        self.features = np.zeros((len(self.review_ints), config.SEQ_LENGTH), dtype=int)
        for i, row in enumerate(self.review_ints):
            self.features[i, -len(row):] = np.array(row)[:config.SEQ_LENGTH]

    def process_new_instance(self, input):
        input = str(input).lower()
        input =''.join([x for x in input if x not in punctuation])
        encoded_input =[]
        for token in input.split():
            encoded_input.append(self.vocab_to_int[token])
        padded_input = np.zeros((1, config.SEQ_LENGTH), dtype=int)
        padded_input[0, -len(encoded_input):] = np.array(encoded_input)[0:config.SEQ_LENGTH]
        return padded_input







