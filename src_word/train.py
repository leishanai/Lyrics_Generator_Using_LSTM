import os
import re
import numpy as np
import unicodedata
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
# disable tensorflow warnings
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
'''
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''


class word_model():

    def __init__(self, file_path, length, test_size, epoch, batch_size):
        
        self.file_path = file_path
        self.epoch = epoch
        self.batch_size = batch_size
        self.length = length        
        self.test_size = test_size
        
    def train(self):
        
        self.data_preparation()
        self.rnn()
        self.fit()

    def resume(self):
        '''
        load pretrained model, dict and cleaned data
        continue to train
        '''
        self.retrieve()
        print(self.model.summary())
        self.fit()

    def data_clean(self):
        '''
        we used tokenizer from keras to integer encode unique words
        '''

        # read data from file and cut file into sequences with fixed length
        with open(self.file_path, 'r') as f:
            raw_text = f.read()
            raw_text = decontracted(raw_text) # expand contractions
            doc = normalizeString(raw_text).split() # doc consists of separate tokens
            # cut encoded doc into 'self.length+1' long pairs
            pairs = [' '.join(doc[i-10:i+1]) for i in range(10,len(doc))] # join back to str

        # use Tokenizer from keras to encode data
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(pairs) # get index for words
        # e.g. for sequences=[['hello there', 'how are you']], sequences = [[1,2], [4,5,6]]
        encoded_pairs = self.tokenizer.texts_to_sequences(pairs) # integer encode
        # vocabulary size
        self.vocab_size = len(self.tokenizer.word_index) + 1 # +1 as index starts from 1(not 0)
        # ix2word -> tokenizer.index_word, word2ix -> tokenizer.word_index
        dump(self.tokenizer, open('save/tokenizer.pkl', 'wb'))
        
        return encoded_pairs


    def data_preparation(self):

        pairs = self.data_clean()
        # separate each line into input and output
        pairs = np.array(pairs)
        # X has length long characters, y is the last character
        X, y = pairs[:,:-1], pairs[:,-1] # X: 10words, y: 1 word
        # one hot encoding targets to fit 'categorical_crossentropy'
        y = to_categorical(y, num_classes=self.vocab_size)
        # train/test split
        self.X_train, self.X_test, self.y_train, self.y_test =\
            train_test_split(X, y, test_size=test_size, random_state=666)
        # save data in a npz file
        np.savez_compressed('save/data.npz',
            x_train = self.X_train, x_test = self.X_test, y_train = self.y_train, y_test = self.y_test)

    def embeds(self):
        '''
        match pretrained embedding and dataset
        embedding -> unique_words x emb_size
        '''
        # load glove
        glove_path = '../data/glove.6B.50d.txt'
        EMBEDDING_DIM = 50
        glove = pretrained(glove_path)
        # initialize embedding matrix for our dataset
        embedding_matrix = np.zeros((self.vocab_size, EMBEDDING_DIM))
        # count words that appear only in the dataset. word_index.items() yields dict of word:index pair
        for word, ix in self.tokenizer.word_index.items():
            embedding_vector = glove.get(word)
            if embedding_vector is not None:
                # words not found in glove matrix will be all-zeros.
                embedding_matrix[ix] = embedding_vector

        return embedding_matrix, glove

    def rnn(self):
        
        embedding_matrix, _ = self.embeds()

        self.model = Sequential()
        # embedding layer, remove weights to disable using pretrained word embedding
        self.model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
        weights=[embedding_matrix], input_length=self.length, trainable=False))
        # True: return c_t and h_t to next layer
        self.model.add(LSTM(200, return_sequences=True, kernel_initializer='he_normal'))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(200, kernel_initializer='he_normal'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(300, activation='relu', kernel_initializer='he_normal'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(embedding_matrix.shape[0], activation='softmax'))
        # setting up parameters
        self.model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
        print(self.model.summary())

    def fit(self):
        
        # recording loss history
        history = LossHistory()
        # save the model (not just weights) after each epoch
        weights = ModelCheckpoint(filepath = 'save/model.h5')
        # training
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test),
            batch_size=self.batch_size, epochs=self.epoch, callbacks=[history, weights])

    def retrieve(self):
        # load pretrained model
        self.model = load_model('save/model.h5')
        # load cleaned data
        data = np.load('save/data.npz')
        self.X_train, self.X_test = data['x_train'], data['x_test']
        self.y_train, self.y_test = data['y_train'], data['y_test']



######################################################################################################
# helper functions
# customize an History class that save losses to a file for each epoch
class LossHistory(Callback):
    
    def on_train_begin(self, logs=None):
        
        if os.path.exists('save/loss.npz'):
            self.loss_array = np.load('save/loss.npz')['loss']
        else:
            self.loss_array = np.empty([2,0])
    
    def on_epoch_end(self, epoch, logs=None):
        # append new losses to loss_array
        loss_train = logs.get('loss')
        loss_test = logs.get('val_loss')

        loss_new = np.array([[loss_train], [loss_test]]) # 2 x 1 array
        self.loss_array = np.concatenate((self.loss_array, loss_new), axis=1)
        # save to disk
        np.savez_compressed('save/loss.npz', loss=self.loss_array)

# expand contractions
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    
    return phrase

# remove punctuations
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z]+", r" ", s)
    return s

# pretrained word embedding
def pretrained(glove_path):
    '''
    load pretrained glove and return it as a dictionary
    '''
    # dimension of import word2vec file
    glove = {}
    with open(glove_path,'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            glove[word] = coefs
    return glove



if __name__ == "__main__":
    
    file_path = "../data/eminem.txt"
    # hyperparameters
    length = 10 # fixed length of input sequences
    epoch = 2
    batch_size = 125
    test_size = 0.33 
    # instantiate the model
    model = word_model(file_path, length, test_size, epoch, batch_size)
    # train or resume train
    if not os.path.exists('save/model.h5'):
        print('<==========| Data preprocessing... |==========>')
        model.train()
    
    else:
        print('<==========| Resume from last training... |==========>')
        model.resume()