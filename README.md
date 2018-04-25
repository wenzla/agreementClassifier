# Introduction

This is a simple project that uses a subsection of the IAC (Internet Agreement Corpus) and a Naive Bayes algorithm to create a classifier that classifies responses that agree or disagree with a previous comment.

As stated, the current implementation uses Naive Bayes with a bag of words assumption. There are some slight modifications to note. Stems are used instead of words through the LancasterStemmer implementation on NLTK. Bigrams are also used as features to get a small context window. Finally note not all words are used, only a beginning substring of words and bigrams are considered.

This implementation has about 73.42% accuracy.

There is now a neural network implementation of the IAC classifier using bag of words (but with LancasterStemmer stems instead of words) as the feature vector.  It is trained on 'strong' arguments

The neural network has around 99% accuracy if testing on the 'strong' arguments (confidence of 1.0 to 5.0 or -1.0 to -5.0).
The accuracy drops down to ~86% percent if used on any statement.

# How to run

The application uses python 3 for the Naive Bayes Classifier and python 2 for the neural network.

To run the application download the repo and make sure you have the following python libraries:

| Python Library  	| 2.7 	| 3.4 	|
|-----------------	|-----	|-----	|
| nltk            	| X   	| X   	|
| math            	| X   	|     	|
| collections     	| X   	|     	|
| counter         	| X   	|     	|
| re              	| X   	| X   	|
| csv             	| X   	| X   	|
| argparse        	| X   	| X   	|
| gensim          	| X   	| X   	|
| numpy           	| X   	| X   	|
| os              	|     	| X   	|
| json            	|     	| X   	|
| time            	|     	| X   	|
| datetime        	|     	| X   	|
| sys             	|     	| X   	|
| matplotlib        | X     |       |
| scikit            | X     |       |

Next, navigate to the agreementClassifier directory of the application.

Both classifiers use the same arguments: classifier.py [training set csv] [testing set csv]

Examples shown below:

>> python3 src\main.py data\iac-b-train.csv data\iac-b-dev.csv

and 

>> python src\NNClassify.py data\iac-b-train.csv data\iac-b-dev.csv

also I use the scikit python library to check performance against already implemented classifiers


## TODOs

* Use entirety of IAC rather than smaller balanced set. (Not on neural network as the training already took long using a small subset of the data)
* Try to weasel out a little more performance out of the custom classifiers.


Enjoy!

-- Allen Wenzl
