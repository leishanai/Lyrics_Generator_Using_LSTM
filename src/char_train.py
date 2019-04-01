import numpy as np
import os
from pickle import dump
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
# from keras.callbacks import TensorBoard
# from importance_sampling.training import ImportanceTraining


class char_model():

    def __init__(self, file_path, length, test_size, epoch, batch_size, trained_model, path_X_train, path_X_test, path_y_train, path_y_test):
        self.file_path = file_path
        self.epoch = epoch
        self.batch_size = batch_size
        self.length = length        
        self.test_size = test_size
        # from pretrained model
        self.trained_model = trained_model
        self.path_X_train = path_X_train
        self.path_X_test = path_X_test
        self.path_y_train = path_y_train
        self.path_y_test = path_y_test
        
    def train(self):
        self.data_clean()
        self.one_hot()
        self.data_preparation()
        self.rnn()
        self.fit()
        self.save()

    def resume(self):
        # load pretrained model and cleaned data
        self.retrieve()
        self.fit()
        self.save()

    def data_clean(self):
        # read data from file and integer-encode the data
        with open(self.file_path, 'r') as f:
            self.raw_text = f.read()
            # integer-encoding chars
            chars = sorted(list(set(self.raw_text))) # unique chars
            self.char2ix = dict((c, i) for i, c in enumerate(chars)) # char:index
            self.ix2char = dict((i, c) for i, c in enumerate(chars)) # index:char
            self.vocab_size = len(self.char2ix)

    def one_hot(self):
        # arange raw_text into sequences
        self.sequences = list()
        for i in range(self.length, len(self.raw_text)):
            seq = self.raw_text[i-self.length:i+1] # cut a 'length+1' long string
            encoded_seq = [self.char2ix[char] for char in seq] # list of list
            self.sequences.append(encoded_seq)
    
    def data_preparation(self):
        # separate each line into input and output
        self.sequences = np.array(self.sequences)
        # X has length long characters, y is the last character
        X, y = self.sequences[:,:-1], self.sequences[:,-1]
        # one-hot encoding data
        sequences = [to_categorical(x, num_classes=self.vocab_size) for x in X] 
        X = np.array(sequences)
        y = to_categorical(y, num_classes=self.vocab_size)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=666)

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
        # let the fun begin
        self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), batch_size=batch_size, epochs=epoch)

    def save(self):
        self.model.save('../save/char_model.h5')
        # save embedding
        try:
            open('../save/char2ix.pkl', 'rb')
        except FileNotFoundError:
            dump(self.char2ix, open('../save/char2ix.pkl', 'wb'))
        try:
            open('../save/ix2char.pkl', 'rb')
        except FileNotFoundError:
            dump(self.ix2char, open('../save/ix2char.pkl', 'wb'))
        # save data
        if not os.path.exists(self.path_X_train):
            np.save(self.path_X_train, self.X_train)
            
        if not os.path.exists(self.path_X_test):
            np.save(self.path_X_test, self.X_test)

        if not os.path.exists(self.path_y_train):
            np.save(self.path_y_train, self.y_train)
            
        if not os.path.exists(self.path_y_test):
            np.save(self.path_y_test, self.y_test)

    def retrieve(self):
        # load data, pretrained model and one-hot embedding
        self.model = load_model(self.trained_model)
        self.X_train, self.y_train = np.load(self.path_X_train), np.load(self.path_y_train)
        self.X_test, self.y_test = np.load(self.path_X_test), np.load(self.path_y_test)

if __name__ == "__main__":
    # fresh train
    file_path = "../data/eminem.txt"
    length = 10
    epoch = 1
    batch_size = 125
    test_size = 0.33
    # model and data from pretrained model
    trained_model='../save/char_model.h5'
    path_X_train, path_y_train = '../save/X_train.npy', '../save/y_train.npy'
    path_X_test, path_y_test = '../save/X_test.npy', '../save/y_test.npy'
    model = char_model(file_path, length, test_size, epoch, batch_size, trained_model, path_X_train, path_X_test, path_y_train, path_y_test)
    #model.train()
    # train a pretrained model
    model.resume()
