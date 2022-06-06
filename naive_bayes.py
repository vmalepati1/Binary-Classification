from keras.datasets import imdb
import numpy as np

from sklearn.naive_bayes import MultinomialNB

from dataset import get_processed_data

word_index = imdb.get_word_index()

x_train, y_train, x_test, y_test = get_processed_data()

print('Train data shape: ' + str(x_train.shape))
print('Train labels shape: ' + str(y_train.shape))

nb = MultinomialNB().fit(x_train, y_train)

print('Score on train: ' + str(nb.score(x_train, y_train)))
print('Score on test: ' + str(nb.score(x_test, y_test)))

example_review = 'loved the plot'
words = example_review.split(' ')

indices = [(word_index[word] + 3) for word in words]

# print(indices)
# indices = imdb_train_data[5]

validation_sample = np.zeros((1, 10000))
validation_sample[0, indices] = 1.

print('Validation sample review:')
print(example_review)
print('Classification: ' + str(nb.predict(validation_sample)[0]))
