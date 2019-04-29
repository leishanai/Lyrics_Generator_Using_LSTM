# Word-level language model

This is many-to-one model (input is a sequence and output is a word).

Used pretrained embedding matrix from [GloVe](https://nlp.stanford.edu/projects/glove/)

## How to use the model?

* Run ```train.py``` to train the model.
* Run ```generator.py ``` to generate lyrics.
* Model and cleaned data will be stored in save folder in the first run. Model will be saved at the end of each epoch.

## How to tweak the model?

* Tune hyperparameter: batch_size, epoch, length(max_len of sequences), test_size.
* Tweak rnn function: add more layers, LSTM -> GRU, change number of memory units of LSTM and etc.
* Train word embedding by yourself.