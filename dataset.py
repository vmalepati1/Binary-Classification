from keras.datasets import imdb
import numpy as np

# Convert list of sequences into one-hot encoding
def vectorize_data(seqs, dim):
    results = np.zeros((len(seqs), dim))
    for i, seq in enumerate(seqs):
        results[i, seq] = 1.
    return results

def get_processed_data():
    (imdb_train_data,imdb_train_labels),(imdb_test_data,imdb_test_labels) = imdb.load_data(num_words=10000)

    x_train = vectorize_data(imdb_train_data, 10000)
    x_test = vectorize_data(imdb_test_data, 10000)

    y_train=np.asarray(imdb_train_labels).astype('float32')
    y_test=np.asarray(imdb_test_labels).astype('float32')

    return x_train, y_train, x_test, y_test
