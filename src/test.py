import gensim
import argparse
import csv
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import numpy as np
from gensim.models.word2vec import Word2Vec
from collections import Counter, defaultdict
from meanvectorizer import MeanVectorizer
from tfidfvectorizer import TfIdfW2Vectorizer

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

mult_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
bern_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
mult_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
bern_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
svc = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])
svc_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])
etree_w2v = Pipeline([("word2vec vectorizer", MeanVectorizer(w2v)), ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_w2v_tfidf = Pipeline([("word2vec vectorizer", TfIdfW2Vectorizer(w2v)), ("extra trees", ExtraTreesClassifier(n_estimators=200))])

etree_w2v_tfidf.fit(data, ratings)
etree_w2v.fit(data, ratings)
mult_nb.fit(data, ratings)
bern_nb.fit(data, ratings)
mult_nb_tfidf.fit(data, ratings)
bern_nb_tfidf.fit(data, ratings)
svc.fit(data, ratings)
svc_tfidf.fit(data, ratings)

with open(args.test, 'r') as csv_test:
    train_reader = csv.reader(csv_test, delimiter=',')
    next(train_reader)
    for row in train_reader:
        rating = float(row[1])
        if rating >= -1 and rating < 1:
            continue       
        data.append(getWords(row[3]))
        ratings.append(int(rating))
        classes.append(classer(row))
#End with



'''
#words most similar to the word in parenthesis
print model.most_similar('disagree')
print model['agree']
'''