# Word-level language model

This is seq2seq model (input and output are both sequence).

Used pretrained embedding matrix from [GloVe](https://nlp.stanford.edu/projects/glove/)

## How to use the model?

* Run ```train.py``` to train the model.
* Run ```generator.py ``` to generate lyrics.
* Used Keras functional API to build the model. There are three 'Model' defined in model function:
	a. self.model -> seq2seq model
	b. self.encoder_model -> encoder part, self.decoder_model -> same as encoder part of self.model but with different output.
	c. inference model = self.encoder_model + self.decoder_model
* Model and cleaned data will be stored in save folder in the first run. Model will be saved at the end of each epoch.

## How to tweak the model?

* Tune hyperparameter: batch_size, epoch, test_size.
* Tweak model function: add more layers, LSTM -> GRU, change number of memory units of LSTM and etc.
* Model used teacher forcing for training process. Implement exploitation could improve the performance.
* Truncate sequences by setting up min- and max-length cutoffs.
* Implement unknown tokens for rare words.
* Make a smarter 'decontracted' function to expand abbreviations.
* Train word embedding by yourself.