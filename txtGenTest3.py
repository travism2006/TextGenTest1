"""
    From https://towardsdatascience.com/how-our-device-thinks-e1f5ab15071e
"""


import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
#--------------------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

from datetime import datetime
now = datetime.now()
print("now =", now)
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)

import time
time.sleep(5)  # sleep for 1sec


tokenizer = Tokenizer() #instantiating the tokenizer
sentence = open("Story 1945 Scenario 1 Script 1.csv").read()
import re

corpus = sentence.lower().split("\n") #converting the sentence to lowercase
finalCorpus = None
print("len of corpus=",len(corpus))
#print(corpus)
#for c in corpus:
#        print(c)
#time.sleep(10)


tokenizer.fit_on_texts(corpus) #creates tokens for each words 
total_words = len(tokenizer.word_index) + 1 #calculating total number of words in the initial sentence

input_sequences = [] #training features (x) will be a list

for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0] #converts each sentence as its tokenized equivalent
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1] #generating n gram sequences
		input_sequences.append(n_gram_sequence) #appending each n gram sequence to the list of our features (xs)


max_sequence_len = max([len(x) for x in input_sequences]) #calculating the length of the longest sequence
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')) #pre-pading each value of the input_sequence
xs, labels = input_sequences[:,:-1],input_sequences[:,-1] #creating xs and their labels using numpy slicing
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words) #creating one hot encoding values

# -----MODEL-----
model = Sequential() #creating a sequential model
model.add(Embedding(total_words, 64, input_length=max_sequence_len-1)) #adding an embedding layer with 64 as the embedding dimension
model.add(Bidirectional(LSTM(20))) #adding 20 LSTM units
#model.add(Dropout(0.2)) # custom
model.add(Dense(total_words, activation='softmax')) #creating a dense layer with 54 output units (total_words) with softmax activation

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #compiling the model with adam optimiser
history = model.fit(xs, ys, epochs=400, verbose=1) #training for 500 epochs = original


#predicting the next word using an initial sentence
input_phrase = input('Enter your input phrase: ')
predOutput = ""
next_words = 20
  
for _ in range(next_words):
	token_list = tokenizer.texts_to_sequences([input_phrase])[0] #converting our input_phrase to tokens and excluding the out of vcabulary words
	token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre') #padding the input_phrase

	#deprecated as of 10 January 2021 model.predict_classes(token_list, verbose=0) #predicting the token of the next word using our trained model
	predicted = np.argmax(model.predict(token_list), axis=-1)
	output_word = "" #initialising output word as blank at the beginning
	for word, index in tokenizer.word_index.items():
		if index == predicted:
			output_word = word #converting the token back to the corresponding word and storing it in the output_word
			break
	input_phrase += " " + output_word
	predOutput += " " + output_word
#print(input_phrase)
print(predOutput)
