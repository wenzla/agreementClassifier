import gensim
import argparse
import csv
import re
import random
from random import randrange
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np
from gensim.models.word2vec import Word2Vec
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

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
    if score > 0.1 and score <= 5:
        return AGREE_CLASS
    elif score < -0.1 and score >= -5:
        return DISAGREE_CLASS
#End def

def getCorrectRatio(model, data, classes):
    correct, incorrect = (0,0)
    results = model.predict(data)
    for i in range(0,len(classes)):
        if classes[i] == results[i]:
            correct += 1
        else:
            incorrect += 1 
    #end for
    return (correct/float(correct + incorrect))
#end def

def accuracy(data, classes):
    print 'creating classifiers...'
    # Create pipelines for different classifiers where the count vectorizer and the tf-idf vectorizer
    mult_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
    bern_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
    mult_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
    bern_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
    svc = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])
    svc_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])
    etree_w2v = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("extra trees", ExtraTreesClassifier(n_estimators=200))])
    etree_w2v_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("extra trees", ExtraTreesClassifier(n_estimators=200))])
    NN = Pipeline([("word2vec vectorizer", CountVectorizer(analyzer=lambda x: x)), ("NN", MLPClassifier())])
    NN_tfidf = Pipeline([("word2vec vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("NN", MLPClassifier())])
    etree_w2v_tfidf.fit(data, classes)
    etree_w2v.fit(data, classes)
    mult_nb.fit(data, classes)
    bern_nb.fit(data, classes)
    mult_nb_tfidf.fit(data, classes)
    bern_nb_tfidf.fit(data, classes)
    svc.fit(data, classes)
    svc_tfidf.fit(data, classes)
    NN.fit(data, classes)
    NN_tfidf.fit(data, classes)

    with open(args.test, 'r') as csv_test:
        train_reader = csv.reader(csv_test, delimiter=',')
        next(train_reader)
        for row in train_reader:
            rating = float(row[1])
            if rating >= -1 and rating < 1:
                continue       
            data.append(getWords(row[3]))
            classes.append(classer(row))
    #End with
    
    accuracies = []
    accuracies.append(getCorrectRatio(mult_nb, data, classes))
    accuracies.append(getCorrectRatio(mult_nb_tfidf, data, classes))
    accuracies.append(getCorrectRatio(bern_nb, data, classes))
    accuracies.append(getCorrectRatio(bern_nb_tfidf, data, classes))
    accuracies.append(getCorrectRatio(svc, data, classes))
    accuracies.append(getCorrectRatio(svc_tfidf, data, classes))
    accuracies.append(getCorrectRatio(etree_w2v, data, classes))
    accuracies.append(getCorrectRatio(etree_w2v_tfidf, data, classes))
    accuracies.append(getCorrectRatio(NN, data, classes))
    accuracies.append(getCorrectRatio(NN_tfidf, data, classes))
    return accuracies   
#end def

def getRandomIndexes(num, length):
    indexes = []
    for i in range(num):
        randomIndex = randrange(length)
        if randomIndex in indexes:
            i = i - 1
        else:
            indexes.append(randomIndex)
        #end if
    #end for
    return indexes
#end def

def getRandItems(indexes, items):
    itemList = []
    for i in range(len(indexes)):
        itemList.append(items[indexes[i]])
    #end for
    return itemList
#end def

def graphStuff(XVals, graphProbs):
    #Get inverse of the graphProbs matrix
    graphProbsI = []
    for i in range(len(graphProbs[0])):
        tempArray = []
        for k in range(len(graphProbs)):
            tempArray.append(graphProbs[k][i])
        #end for
        graphProbsI.append(tempArray)
    #end for
    print graphProbsI

    for i in range(len(graphProbsI)):
        plt.plot(XVals, graphProbsI[i])    
    #end for
    legendVals = ['MultinomialNB', 'MultinomialNB tf-idf', 'BernoulliNB', 'BernoulliNB tf-idf', 'SVC', 'SVC tf-idf', 'Forest', 'Forest tf-idf', 'NN', 'NN-tf-idf']
    plt.legend(legendVals, loc='lower right')
    plt.xlabel('Size of Training Set', fontsize=16)
    plt.ylabel('Probability', fontsize=16)
    plt.show()
#end def

parser = argparse.ArgumentParser()
parser.add_argument('train', help='The filename that points to training set.')
parser.add_argument('test', help='The filename that points to test set.')
args = parser.parse_args()
data = []
classes = []

with open(args.train, 'r') as csv_train:
    train_reader = csv.reader(csv_train, delimiter=',')
    next(train_reader)
    for row in train_reader:
        rating = float(row[1])
        if rating >= -1 and rating < 1:
            continue       
        data.append(getWords(row[3]))
        classes.append(classer(row))
#End with

graphProbs = []
num_to_select = 24
XVals = []

while num_to_select < len(data):
    random_indexes = getRandomIndexes(num_to_select, len(data))
    random_items = getRandItems(random_indexes, data)
    random_classes = getRandItems(random_indexes, classes)
    probs = accuracy(random_items, random_classes)
    XVals.append(num_to_select)
    graphProbs.append(probs)
    print 'Multinomial NB classification accuracy: {}'.format(probs[0])
    print 'Multinomial NB tf-idf classification accuracy: {}'.format(probs[1])
    print 'Bernoulli NB classification accuracy: {}'.format(probs[2])
    print 'Bernoulli NB tf-idf classification accuracy: {}'.format(probs[3])
    print 'SVC classification accuracy: {}'.format(probs[4])
    print 'SVC tf-idf classification accuracy: {}'.format(probs[5])
    print 'Forest classification accuracy: {}'.format(probs[6])
    print 'Forest tf-idf classification accuracy: {}'.format(probs[7])
    print 'NN classification accuracy: {}'.format(probs[8])
    print 'NN tf-idf classification accuracy: {}'.format(probs[9])
    num_to_select = num_to_select * 2
#end while

#print XVals
#print graphProbs
graphStuff(XVals, graphProbs)

