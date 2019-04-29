from __future__ import print_function # from __future__ imports must occur at the beginning of the file
import os
import re
import unicodedata
import numpy as np
from pickle import dump
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import Input, Dense, Embedding, LSTM
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
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



class seq2seq_model():

    def __init__(self, file_path, unit_encoder, unit_decoder, test_size, epoch, batch_size):
        
        self.file_path = file_path
        self.epoch = epoch
        self.batch_size = batch_size
        self.unit_encoder = unit_encoder
        self.unit_decoder = unit_decoder       
        self.test_size = test_size
        
    def train(self):
        
        self.data_preparation()
        self.model()
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
        read data from file and integer-encode the data by using tokenizer
        '''
        
        lines = open(self.file_path, encoding='utf-8').read().strip().split('\n') # read files line by line
        lines = [decontracted(line) for line in lines] # expand contractions
        lines = [normalizeString(line) for line in lines] # clean up text
        # add EOS and SOS tokens
        pairs = [[lines[i]+' EOS', 'SOS '+lines[i+1]+' EOS'] for i in range(0,len(lines)-1)]
        # use Tokenizer from keras to encode data
        self.tokenizer = Tokenizer()
        lines.append('EOS SOS') # add EOS and SOS token
        self.tokenizer.fit_on_texts(lines) # integer encoding       
        encoded_pairs = [self.tokenizer.texts_to_sequences(pair) for pair in pairs] # convert strings to index
        # vocabulary size
        self.vocab_size = len(self.tokenizer.word_index) + 1 # number of distinct words. +1 as index starts from 1(not 0)
        # ix2word -> tokenizer.index_word, word2ix -> tokenizer.word_index
        dump(self.tokenizer, open('save/tokenizer.pkl', 'wb'))
        
        return encoded_pairs


    def data_preparation(self):

        encoded_pairs = self.data_clean()      
        data_size = len(encoded_pairs)
        # sort out max lengths of input data and target data
        max_len_x, max_len_y = 0, 0
        for pair in encoded_pairs:      
            max_len_x = max( len(pair[0]), max_len_x)
            max_len_y = max( len(pair[1])-1, max_len_y)
        # Note that input data must be 2D before feeding to embedding layer. Padding data.
        self.encoder_input = np.zeros((data_size, max_len_x), dtype='float32')
        self.decoder_input = np.zeros((data_size, max_len_y), dtype='float32')
        self.decoder_target = np.zeros((data_size, max_len_y, self.vocab_size), dtype='float32')       
        # filling data into frames
        for row, pair in enumerate(encoded_pairs):            
            for col, token_ix in enumerate(pair[0]):
                self.encoder_input[row, col] = token_ix # encoder input
            # decoder_target_data is ahead of decoder_input_data by one timestep
            for col, token_ix in enumerate(pair[1]):
                if col < len(pair[1])-1:
                    self.decoder_input[row, col] = token_ix # encoder_input excludes EOS token
                if col > 0:
                    self.decoder_target[row, col - 1, token_ix] = 1. # # decoder_target excludes SOS token
        
        # save data in a npz file
        np.savez_compressed('save/data.npz', encoder_input = self.encoder_input,
            decoder_input = self.decoder_input, decoder_target = self.decoder_target)
      

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


    def model(self):
        '''
        this function defines seq2seq model and inference model for prediction
        self.model(seq2seq model): encoder_inputs + decoder_inputs -> decoder_output
        self.encoder_model: encoder_intputs -> encoder_states
        self.decoder_model: decoder_inputs + encoder_states -> decoder_output
        inference mode = self.encoder_model + self.decoder_model
        '''

        # load pretrained embedding
        embedding_matrix, _ = self.embeds()
        Shared_Embedding = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
        weights=[embedding_matrix], trainable=False)
        ######################################################################################
        # encoder_input -> embedding
        encoder_inputs = Input(shape=(None,))
        embedded_encoder_inputs = Shared_Embedding(encoder_inputs)
        # embedding -> hidden states
        encoder = LSTM(self.unit_encoder, return_state=True)
        _, state_h, state_c = encoder(embedded_encoder_inputs) # we don't care about encoder outputs
        encoder_states = [state_h, state_c] # concatenate two hiddens states
        # encoder_model instance
        self.encoder_model = Model(encoder_inputs, encoder_states)
        # print(self.encoder_model.summary())
        ######################################################################################
        # decoder_input -> embedding
        decoder_inputs = Input(shape=(None,))
        embedded_decoder_inputs = Shared_Embedding(decoder_inputs)
        # embedding -> hidden states
        decoder_lstm = LSTM(self.unit_decoder, return_sequences=True, return_state=True)
        y, _, _ = decoder_lstm(embedded_decoder_inputs, initial_state=encoder_states)
        # softmax layer
        decoder_dense = Dense(self.vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(y)
        ######################################################################################
        # the whole seq2seq model
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        print(self.model.summary())
        ######################################################################################
        # inference mode
        decoder_state_input_h = Input(shape=(unit_decoder,))
        decoder_state_input_c = Input(shape=(unit_decoder,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        embedded = Shared_Embedding(decoder_inputs) # vectorize decoder_inputs

        decoder_outputs2, state_h2, state_c2 = decoder_lstm(embedded, initial_state=decoder_states_inputs)
        decoder_states2 = [state_h2, state_c2]
        decoder_outputs2 = decoder_dense(decoder_outputs2)
        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                            [decoder_outputs2] + decoder_states2)
        # print(self.decoder_model.summary())



    def fit(self):

        # recording loss history
        history = LossHistory()
        # save the model (not just weights) after each epoch
        weights = ModelCheckpoint(filepath = 'save/model.h5')
        # training
        self.model.fit([self.encoder_input, self.decoder_input], self.decoder_target, batch_size=self.batch_size,
                epochs=self.epoch, validation_split=self.test_size, callbacks=[history, weights])
        
        if  os.path.exists('save/model.h5'):

            lstm_encoder = self.model.layers[3].get_weights()
            lstm_decoder = self.model.layers[4].get_weights()
            dense_decoder = self.model.layers[5].get_weights()

            self.encoder_model.layers[2].set_weights(lstm_encoder)
            self.decoder_model.layers[4].set_weights(lstm_decoder)
            self.decoder_model.layers[5].set_weights(dense_decoder)
        
        # save these for evaluation. Note that they will not saved if training is interrupted
        self.encoder_model.save('save/encoder.h5')
        self.decoder_model.save('save/decoder.h5')
        '''
        saving model causes following error, not sure what this error is or how to resolve it...

        /home/leishan/anaconda3/envs/dlpy36/lib/python3.6/site-packages/keras/engine/network.py:877: 
        UserWarning: Layer lstm_2 was passed non-serializable keyword arguments: {'initial_state': 
        [<tf.Tensor 'lstm_1/while/Exit_2:0' shape=(?, 150) dtype=float32>, 
        <tf.Tensor 'lstm_1/while/Exit_3:0' shape=(?, 150) dtype=float32>]}. 
        They will not be included in the serialized model (and thus will be missing at deserialization time).
        '. They will not be included '
        '''

    def retrieve(self):
        # load pretrained model
        self.model = load_model('save/model.h5')
        self.encoder_model = load_model('save/encoder.h5')
        self.decoder_model = load_model('save/decoder.h5')
        # load cleaned data
        data = np.load('save/data.npz')
        self.encoder_input = data['encoder_input']
        self.decoder_input = data['decoder_input']
        self.decoder_target = data['decoder_target']



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
    epoch = 5
    batch_size = 125
    unit_encoder = 150
    unit_decoder = 150
    test_size = 0.33 
    # instantiate the model
    model = seq2seq_model(file_path, unit_encoder, unit_decoder, test_size, epoch, batch_size)
    # train or resume train
    if not os.path.exists('save/model.h5'):
        print('<==========| Data preprocessing... |==========>')
        model.train()
    
    else:
        print('<==========| Resume from last training... |==========>')
        model.resume()

################################ plot the structure of the model #############################

# plot_model(model, show_shapes=True, to_file="../images/model.jpg")
# plot_model(encoder_model, show_shapes=True, to_file="../images/encoder.jpg")
# plot_model(decoder_model, show_shapes=True, to_file="../images/decoder.jpg")

##############################################################################################

