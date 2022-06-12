from keras.datasets import imdb
import numpy as np

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from dataset import get_processed_data

from numpy import random

word_index = imdb.get_word_index()

x_train, y_train, x_test, y_test = get_processed_data()

print('Train data shape: ' + str(x_train.shape))
print('Train labels shape: ' + str(y_train.shape))

random.seed(1234)

nb = MultinomialNB()
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=30, max_depth=9).fit(x_train, y_train)
svm = LinearSVC(C=0.0001)

evc=VotingClassifier(estimators=[('mnb',nb),('lr',lr),('rf',rf),('svm',svm)],voting='hard')
evc.fit(x_train, y_train)

print('Score on train: ' + str(evc.score(x_train, y_train)))
print('Score on test: ' + str(evc.score(x_test, y_test)))

example_review = 'loved the plot'
words = example_review.split(' ')

indices = [(word_index[word] + 3) for word in words]

# print(indices)
# indices = imdb_train_data[5]

validation_sample = np.zeros((1, 10000))
validation_sample[0, indices] = 1.

print('Validation sample review:')
print(example_review)
print('Classification: ' + str(evc.predict(validation_sample)[0]))
