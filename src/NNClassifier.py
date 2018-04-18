import gensim
import argparse
import csv
import re
import random
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
import nltk
import os
import json
import datetime
import sys
import time
import numpy as np
reload(sys)
sys.setdefaultencoding("utf-8")

#The below statment may need to be run if this is the first time running this
#nltk.download('punkt')

AGREE_CLASS = 'AGREE'
DISAGREE_CLASS = 'DISAGREE'

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)
 
def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

def think(sentence, show_details=False):
    x = bow(sentence.lower(), words, show_details)
    if show_details:
        print ("sentence:", sentence, "\n bow:", x)
    # input layer is our bag of words
    l0 = x
    # matrix multiplication of input and hidden layer
    l1 = sigmoid(np.dot(l0, synapse_0))
    # output layer
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2

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

def train(X, y, hidden_neurons=10, alpha=1, epochs=50, dropout=False, dropout_percent=0.5):

    print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else '') )
    print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X),len(X[0]),1, len(classes)) )
    np.random.seed(1)

    last_mean_error = 1
    # randomly initialize our weights with mean 0
    synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2*np.random.random((hidden_neurons, len(classes))) - 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)

    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)
        
    for j in iter(range(epochs+1)):
        # Feed forward through layers 0, 1, and 2
        if j%100 == 0 or j < 100:
            print j
        #end if
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))
                
        if(dropout):
            layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # how much did we miss the target value?
        layer_2_error = y - layer_2

        if (j% 1000) == 0 and j > 5000:
            # if this thousand iteration's error is greater than the last iteration, break out
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print ("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) )
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                break
                
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # how much did each layer1 value contribute to the layer2 error (according to the weights)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # in what direction is the target layer 1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
        
        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))
        
        if(j > 0):
            synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))        
        
        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update
        
        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update

    now = datetime.datetime.now()

    # save synapses to file so it can be easily loaded for later use
    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
              }
    synapse_file = "synapses2.json"

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print ("saved synapses to:", synapse_file)
#End def

def classify(sentence, show_details=False):
    results = think(sentence, show_details)

    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ] 
    results.sort(key=lambda x: x[1], reverse=True) 
    return_results =[[classes[r[0]],r[1]] for r in results]
    #print ("%s \n classification: %s" % (sentence, return_results))
    if len(return_results) < 1:
        rand = random.randint(1,3)
        if rand == 1:
            return classer([rand , -5])
        else:
            return classer([rand , 5])
    #end if
    return return_results[0]
#End def

stemmer = LancasterStemmer()
parser = argparse.ArgumentParser()
parser.add_argument('train', help='The filename that points to training set.')
parser.add_argument('test', help='The filename that points to test set.')
args = parser.parse_args()
data = []
line = []
ratings = []
classes = []

with open(args.train, 'r') as csv_train:
    train_reader = csv.reader(csv_train, delimiter=',')
    next(train_reader)
    for row in train_reader:
        rating = float(row[1])
        if rating >= -0.2 and rating < 0.2:
            continue       
        data.append(getWords(row[3]))
        line.append(row[3])
        ratings.append(int(rating))
        classes.append(classer(row))
#End with

training_data = []

for i in range(0,len(data)):
    training_data.append({"class":classes[i], "sentence":line[i]})
#end for

words = []
classes = []
statements = []
ignore_words = ['?','.']
# loop through each sentence in the training data
for pattern in training_data:
    # tokenize each word in the sentence
    w = nltk.word_tokenize(pattern["sentence"])
    # add to the dictionary
    words.extend(w)
    # add to statements in the corpus
    statements.append((w, pattern['class']))
    # add to the classes list
    if pattern['class'] not in classes:
        classes.append(pattern['class'])
#End for
# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = list(set(words))

# remove duplicates
classes = list(set(classes))

# create the training data
training = []
output = []
# create an empty array for the output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for statement in statements:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = statement[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    training.append(bag)
    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(statement[1])] = 1
    output.append(output_row)

X = np.array(training)
y = np.array(output)

# DO THIS IF YOU WANT TO TRAIN ANOTHER NN
# train(X, y, hidden_neurons=20, alpha=0.1, epochs=10000, dropout=False, dropout_percent=0.2)

print "Done processing"

# probability threshold
ERROR_THRESHOLD = 0.2
# load our calculated synapse values
synapse_file = 'synapses2.json' 
with open(synapse_file) as data_file: 
    synapse = json.load(data_file) 
    synapse_0 = np.asarray(synapse['synapse0']) 
    synapse_1 = np.asarray(synapse['synapse1'])
#End with

dataTest = []
ratingsTest = []
classesTest = []
resultTest = []

print "Classifying test results..." 

with open(args.test, 'r') as csv_test:
    test_reader = csv.reader(csv_test, delimiter=',')
    next(test_reader)
    for row in test_reader:
        rating = float(row[1])
        if rating >= -1 and rating < 1:
            continue     
            
        dataTest.append(getWords(row[3]))
        resultTest.append(classify(row[3])[0])
        ratingsTest.append(int(rating))
        classesTest.append(classer(row))
#End with

correct, incorrect = (0,0)
for i in range(0,len(classesTest)):
    if classesTest[i] == resultTest[i]:
        correct += 1
    else:
        incorrect += 1
    #End if
#End for

print 'NN classification accuracy: {}'.format(correct/float(correct + incorrect))

while (1):
    userArgument = raw_input('Enter an argument to classify (or type \'exit\' to quit): ')
    if userArgument == 'exit' or userArgument == 'Exit':
        print 'Goodbye!'
        break;
    else:
        print 'Your argument was classified as: {}'.format(classify(userArgument)[0])
    #end if
#End while
