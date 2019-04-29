import os
import numpy as np
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
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
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.style.use('ggplot') # plot style
mpl.rcParams["font.size"] = 18 # font size
# plt.switch_backend('agg')



def generator(seed_text, tokenizer, max_len, encoder_model, decoder_model):
    
    ix2word = tokenizer.index_word
    word2ix = tokenizer.word_index
    # encoding seed text
    seed_text = seed_text+' sos'
    encoded_text = tokenizer.texts_to_sequences([seed_text])[0]
    # obtain last encoder_states and decoder_input
    last_states = encoder_model.predict(encoded_text) # encoder_states
    decoder_input = np.zeros((1,1)) # note that input must be 2D
    decoder_input[0, 0] = word2ix['sos'] # decoder_inputs(SOS token)
    
    stop_condition = False
    generated_text = ''
    while not stop_condition:

        output, h, c = decoder_model.predict([decoder_input] + last_states)
        # greedy search, output is a 3d-list of probabilities
        new_ix = np.argmax(output[0, -1, :])
        new_word = ix2word[new_ix]
        generated_text += ' '+ new_word

        # Exit condition: either hit max length or find stop character.
        if (new_word == 'eos' or len(generated_text) > max_len):
            stop_condition = True
    
        decoder_input[0, 0] = new_ix # new input
        last_states = [h, c] # new states

    print('<=== Seed_text: {} ===>'.format(seed_text))
    print('<=== Generated_lyrics: {} ===>\n'.format(generated_text))



def loss_plot(loss_train, loss_test):
    
    plt.rcParams["axes.grid"] = True
    fig, ax = plt.subplots(figsize=(10,6))
    n = len(loss_train)
    xtcks = np.arange(1, n+1)
    ax.set_title('Learning Curve')
    ax.set_xlabel('Training epochs')
    ax.set_ylabel('Loss')
    ax.plot(xtcks, loss_train, label='Train')
    ax.plot(xtcks, loss_test, label='Test')
    ax.legend()
    plt.savefig('../images/lc_seq2seq.jpg')
    plt.show()



if __name__ == "__main__":

    # Generator
    encoder_model, decoder_model = load_model('save/encoder.h5'), load_model('save/decoder.h5')
    tokenizer = load(open('save/tokenizer.pkl', 'rb'))
    
    max_len = 10

    seed_text = 'who is really slim shady'
    generator(seed_text, tokenizer, max_len, encoder_model, decoder_model)

    seed_text = 'look i was gonna go easy on you'
    generator(seed_text, tokenizer, max_len, encoder_model, decoder_model)



    #############################################################################
    # plot losses
    loss_array = np.load('save/loss.npz')['loss']
    loss_train, loss_test = loss_array[0], loss_array[1]
    loss_plot(loss_train, loss_test)


