import gensim
import argparse
import csv
import re
from nltk.stem.lancaster import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from gensim.models.word2vec import Word2Vec
from collections import Counter, defaultdict
from meanvectorizer import MeanVectorizer
from tfidfvectorizer import TfIdfW2Vectorizer
from gensim.models import KeyedVectors
from nltk.corpus import brown 


AGREE_CLASS = 'AGREE'
DISAGREE_CLASS = 'DISAGREE'

def getWords(line):
    processed = re.sub(r'&#8217;', r"'", line)
    processed = re.sub(r'&#8212;', r"-", line)
    processed = re.sub(r'([^\w\s\'])', r' \1 ', line)
    processed = processed.lower()
    
    return (processed.split())
#End def

def classer(sample):
    """
    Returns the class of a given sample. This is to be used in the Naive Bayes
    classifier. A sample in this case is an item from the IAC data, which has
    been parsed as a csv row.

    Args:
    sample: The sample or data item in IAC. Consists of agreement, quote, and
            response.

    Returns:
    Provides the class of the given sample.
    """
    score = float(sample[1])
    if score > 0.2 and score <= 5:
        return AGREE_CLASS
    elif score < -0.2 and score >= -5:
        return DISAGREE_CLASS
#End def

parser = argparse.ArgumentParser()
parser.add_argument('train', help='The filename that points to training set.')
parser.add_argument('test', help='The filename that points to test set.')
args = parser.parse_args()
data = []
ratings = []
classes = []

with open(args.train, 'r') as csv_train:
    train_reader = csv.reader(csv_train, delimiter=',')
    next(train_reader)
    for row in train_reader:
        rating = float(row[1])
        if rating >= -1 and rating < 1:
            continue       
        data.append(getWords(row[3]))
        ratings.append(int(rating))
        classes.append(classer(row))
#End with

#model = gensim.models.Word2Vec(data, min_count=1)
model = Word2Vec(data, size=100, window=5, min_count=5, workers=2)
w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}


dataTest = []
ratingsTest = []
classesTest = []

with open(args.test, 'r') as csv_test:
    test_reader = csv.reader(csv_test, delimiter=',')
    next(test_reader)
    for row in test_reader:
        rating = float(row[1])
        if rating >= -1 and rating < 1:
            continue       
        dataTest.append(getWords(row[3]))
        ratingsTest.append(int(rating))
        classesTest.append(classer(row))
#End with


'''
#words most similar to the word in parenthesis
print model.most_similar('disagree')
print model['agree']
'''