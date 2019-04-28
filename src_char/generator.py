import os
import numpy as np
from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
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



def generator(model, char2ix, ix2char, input_length, seed_text, output_length):
    
    generated_text = seed_text
    # generate a fixed number of characters
    for _ in range(output_length):
        # integer-encoding seed_text
        encoded = [char2ix[char] for char in generated_text]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=input_length, truncating='pre')[0]
        # one hot encoding
        encoded = to_categorical(encoded, num_classes=len(char2ix))
        # convert to numpy arrary
        encoded = np.array(encoded)
        # input of rnn requires to be tensor 
        encoded = encoded.reshape(1, encoded.shape[0], encoded.shape[1])
        # predict from trained model, greedy search.
        # note that y_ix is nparray, e.g.[1]
        y_ix = model.predict_classes(encoded, verbose=0)
        # boolean of 1==np.array([1]) is true
        generated_text += ix2char[y_ix[0]]
    
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
    plt.savefig('../images/lc_char.jpg')
    plt.show()

# def acc_plot(acc_train, acc_test):
    
#     plt.rcParams["axes.grid"] = True
#     fig, ax = plt.subplots(figsize=(10,6))
#     n = len(acc_train)
#     xtcks = np.arange(1, n+1)
#     ax.set_title('Learning Curve')
#     ax.set_xlabel('Training epochs')
#     ax.set_ylabel('Accuracy')
#     # ax.set_yticklabels(['{:.1%}'.format(x) for x in vals])
#     ax.yaxis.set_major_formatter(FormatStrFormatter('%d%%'))
#     ax.plot(xtcks, acc_train, label='Train')
#     ax.plot(xtcks, acc_test, label='Test')
#     ax.legend()
#     plt.savefig('../images/learning_curve2.jpg')
#     plt.show()

if __name__ == "__main__":

    # Generator
    model = load_model('save/model.h5')
    char2ix, ix2char = load(open('save/dict.pkl', 'rb'))
    
    generator(model, char2ix, ix2char, 10, 'slim shady', 20)
    generator(model, char2ix, ix2char, 10, "i'm rap go", 40)
    generator(model, char2ix, ix2char, 10, 'the way i ', 60)
    generator(model, char2ix, ix2char, 10, 'lose mysel', 80)
    generator(model, char2ix, ix2char, 10, 'not afraid', 100)

    #############################################################################
    # plot losses
    loss_array = np.load('save/loss.npz')['loss']
    loss_train, loss_test = loss_array[0], loss_array[1]
    loss_plot(loss_train, loss_test)
    
    # # plot accuracy, as accuracy seems not to be a proper metric here, I removed it.
    # acc_train = [ 100*acc for acc in history['acc']]
    # acc_test = [ 100*acc for acc in history['val_acc']] 
    # acc_plot(acc_train, acc_test)