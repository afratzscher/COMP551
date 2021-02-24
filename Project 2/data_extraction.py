import tarfile
import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import itertools

np.random.seed(123) 

tf = tarfile.open("aclImdb_v1.tar")
tf.extractall()
print('done extracting')

testfolder = 'aclImdb/test/'
trainfolder = 'aclImdb/train/'

negData = list()
posData = list()
labels = list()

# get train first
# get pos data
for filename in os.listdir(trainfolder+'pos'):
  file = open(trainfolder+'pos/' + filename)
  posData.append(file.read())
  file.close()
# get neg data
for filename in os.listdir(trainfolder+'neg'):
  file = open(trainfolder+'neg/' + filename)
  negData.append(file.read())
  file.close()

trainData = negData + posData

# neg = 0, pos = 1
trainLabel = np.array([0]*len(negData) + [1]*len(posData))


# get test second
negData = list()
posData = list()
# get neg data
for filename in os.listdir(testfolder+'pos'):
  file = open(testfolder+'pos/' + filename)
  posData.append(file.read())
  file.close()
# get pos data
for filename in os.listdir(testfolder+'neg'):
  file = open(testfolder+'neg/' + filename)
  negData.append(file.read())
  file.close()

testData = negData + posData

# neg = 0, pos = 1
testLabel = np.array([0]*len(negData) + [1]*len(posData))

print(len(trainData), len(testData)) 

class bundle:
  def __init__(self, data, target, target_names):
    self.data = data
    self.target = target
    self.target_names = target_names
    self.labels = None

IMDB_train = bundle(trainData, trainLabel, ['neg', 'pos'])
IMDB_test = bundle(testData, testLabel, ['neg', 'pos'])

# have 4 attributes: data, target (0, 1), target_names (neg and pos), and predicted labels

imdb_vectorizer = CountVectorizer()
X_IMDB_train_counts = imdb_vectorizer.fit_transform(IMDB_train.data)
X_IMDB_train_counts.shape

X_IMDB_test_counts = imdb_vectorizer.fit_transform(IMDB_test.data)
X_IMDB_test_counts.shape

# stores text as numbers -> (0,1) 1 means 0th sentence, 1st word/feature index has count 1


imdb_tfidf_transformer = TfidfTransformer()
X_IMDB_train_counts_tfidf = imdb_tfidf_transformer.fit_transform(X_IMDB_train_counts)
  #fit = fit estimator to data
  #transform = transform count amtrix to term frequencies times inverse document frequency
  ## use fit_transform instead of fit and then transform to prevent redundant processing ##
X_IMDB_train_counts_tfidf.shape

#IGNORE THIS: just checked that have the same format :)
# looks like both now have the same format!
from sklearn.naive_bayes import MultinomialNB


clf2 = MultinomialNB().fit(X_IMDB_train_counts_tfidf, IMDB_train.target)
docs_new = ['Horrible movie', 'excellent', 'mediocre']
X_new_counts = imdb_vectorizer.transform(docs_new)
X_new_tfidf = imdb_tfidf_transformer.transform(X_new_counts)

predicted = clf2.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
	print('%r => %s' % (doc, IMDB_train.target_names[category]))
