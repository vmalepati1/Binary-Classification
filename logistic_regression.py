from keras.datasets import imdb
import numpy as np

from sklearn.linear_model import LogisticRegression

(imdb_train_data,imdb_train_labels),(imdb_test_data,imdb_test_labels) = imdb.load_data(num_words=10000)

def vectorize_data(seqs, dim):
    results = np.zeros((len(seqs), dim))
    for i, seq in enumerate(seqs):
        results[i, seq] = 1.
    return results

word_index = imdb.get_word_index()

reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

seq = imdb_train_data[0]

words = ' '.join([reverse_word_index.get(i-3,'?') for i in imdb_train_data[0]])

print('Example review:')
print(words)

print('Classification: ' + str(imdb_train_labels[0]))

x_train = vectorize_data(imdb_train_data, 10000)
x_test = vectorize_data(imdb_test_data, 10000)

print('Train data shape: ' + str(x_train.shape))
print('Train labels shape: ' + str(x_test.shape))

y_train=np.asarray(imdb_train_labels).astype('float32')
y_test=np.asarray(imdb_test_labels).astype('float32')

lr = LogisticRegression(max_iter=1000).fit(x_train, y_train)

print('Score on train: ' + str(lr.score(x_train, y_train)))
print('Score on test: ' + str(lr.score(x_test, y_test)))

example_review = 'loved the plot'
words = example_review.split(' ')

indices = [(word_index[word] + 3) for word in words]

# print(indices)
# indices = imdb_train_data[5]

validation_sample = np.zeros((1, 10000))
validation_sample[0, indices] = 1.

print('Validation sample review:')
print(example_review)
print('Classification: ' + str(lr.predict(validation_sample)[0]))
