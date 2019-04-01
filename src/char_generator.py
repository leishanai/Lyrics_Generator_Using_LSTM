from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np


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
    return generated_text

if __name__ == "__main__":
    model = load_model('../save/char_model.h5')
    char2ix = load(open('../save/char2ix.pkl', 'rb'))
    ix2char = load(open('../save/ix2char.pkl', 'rb'))

    print(generator(model, char2ix, ix2char, 10, 'slim shady', 20))
    print(generator(model, char2ix, ix2char, 10, "i'm rap go", 40))
    print(generator(model, char2ix, ix2char, 10, 'the way i ', 60))
    print(generator(model, char2ix, ix2char, 10, 'lose mysel', 80))
    print(generator(model, char2ix, ix2char, 10, 'not afraid', 100))
