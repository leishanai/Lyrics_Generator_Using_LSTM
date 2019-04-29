import os
import re
import numpy as np
from pickle import dump, load
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, Callback
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


class char_model():

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
        resume training by loading pretrained model and cleaned data
        '''
        self.retrieve()
        print(self.model.summary())
        self.fit()

    def data_clean(self):
        
        # read data from file and integer-encode the data
        with open(self.file_path, 'r') as f:
            self.raw_text = f.read()
            self.raw_text = decontracted(self.raw_text) # expand abbreviation of words
            # integer-encoding chars
            chars = sorted(list(set(self.raw_text))) # unique chars
            char2ix = dict((c, i) for i, c in enumerate(chars)) # char:index
            ix2char = dict((i, c) for i, c in enumerate(chars)) # index:char
            self.vocab_size = len(char2ix)
        # save char2ix and ix2char in one pickle file
        combined = [char2ix, ix2char]
        dump(combined, open('save/dict.pkl', 'wb'))
        return char2ix, ix2char

    def one_hot(self):

        # arange raw_text into sequences and one-hot encode
        char2ix, _ = self.data_clean() 
        sequences = list()
        for i in range(self.length, len(self.raw_text)):
            seq = self.raw_text[i-self.length:i+1] # cut a 'length+1' long string
            encoded_seq = [char2ix[char] for char in seq] # list of list
            sequences.append(encoded_seq)
        return sequences
    
    def data_preparation(self):

        sequences = self.one_hot()
        # separate each line into input and output
        sequences = np.array(sequences)
        # X has length long characters, y is the last character
        X, y = sequences[:,:-1], sequences[:,-1]
        # one-hot encoding data
        sequences = [to_categorical(x, num_classes=self.vocab_size) for x in X] 
        X = np.array(sequences)
        y = to_categorical(y, num_classes=self.vocab_size)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=666)
        # save data in a npz file
        np.savez_compressed('save/data.npz',
            x_train = self.X_train, x_test = self.X_test, y_train = self.y_train, y_test = self.y_test)

    def rnn(self):

        self.model = Sequential()
        # neural network structure
        self.model.add(LSTM(200, input_shape=(self.X_train.shape[1], self.X_train.shape[2]), return_sequences=True, kernel_initializer='he_normal'))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(200, kernel_initializer='he_normal'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(self.vocab_size, activation='softmax', kernel_initializer='he_normal'))
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
            batch_size=batch_size, epochs=epoch, callbacks=[history, weights])

    def retrieve(self):

        # load pretrained model
        self.model = load_model('save/model.h5')
        # load cleaned data
        data = np.load('save/data.npz')
        self.X_train, self.X_test = data['x_train'], data['x_test']
        self.y_train, self.y_test = data['y_train'], data['y_test']



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



if __name__ == "__main__":
    
    file_path = "../data/eminem.txt"
    # hyperparameters
    length = 15 # fixed length of input sequences
    epoch = 2
    batch_size = 125
    test_size = 0.33 
    # instantiate the model
    model = char_model(file_path, length, test_size, epoch, batch_size)
    # train or resume train
    if not os.path.exists('save/model.h5'):
        print('<==========| Data preprocessing... |==========>')
        model.train()
    
    else:
        print('<==========| Resume from last training... |==========>')
        model.resume()