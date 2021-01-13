import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


import numpy
# real code
# all from https://stackabuse.com/text-generation-with-python-and-tensorflow-keras/
import sys
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

file = open("Story 1945 Scenario 1 Script 1.csv").read()
#nltk.download('stopwords')

def tokenize_words(input):
    # lowercase everything to standardize it
    input = input.lower()

    # instantiate the tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)

    # if the created token isn't in the stop words, make it part of "filtered"
    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return " ".join(filtered)

# preprocess the input data, make tokens
processed_inputs = tokenize_words(file)


chars = sorted(list(set(processed_inputs)))
char_to_num = dict((c, i) for i, c in enumerate(chars))


input_len = len(processed_inputs)
vocab_len = len(chars)
print ("Total number of characters:", input_len)
print ("Total vocab:", vocab_len)

seq_length = 150
x_data = []
y_data = []



# loop through inputs, start at the beginning and go until we hit
# the final character we can create a sequence out of
for i in range(0, input_len - seq_length, 1):
    # Define input and output sequences
    # Input is the current character plus desired sequence length
    in_seq = processed_inputs[i:i + seq_length]

    # Out sequence is the initial character plus total sequence length
    out_seq = processed_inputs[i + seq_length]

    # We now convert list of characters to integers based on
    # previously and add the values to our lists
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])

n_patterns = len(x_data)
print ("Total Patterns:", n_patterns)

X = numpy.reshape(x_data, (n_patterns, seq_length, 1))
X = X/float(vocab_len)

y = np_utils.to_categorical(y_data)


# we doin' it!!!
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2)) #article used 0.2
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2)) #article used 0.2
model.add(LSTM(128))
model.add(Dropout(0.2)) #article used 0.2
model.add(Dense(y.shape[1], activation='softmax'))

print("-----model COMPILE----")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



filepath = "model_weights_saved.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
checkpoint2 = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
desired_callbacks = [checkpoint, checkpoint2]



# Fit the model and let it train
print("-----FIT MODEL----")
model.fit(X, y, epochs=75, batch_size=128, callbacks=desired_callbacks)

#----------
import time
startTime = time.time()
#----------

filename = "model_weights_saved.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

endTime = time.time()

print(startTime, ', ', endTime)
print("My program took", time.strftime("%H:%M:%S", time.gmtime(endTime-startTime)), "to run")

num_to_char = dict((i, c) for i, c in enumerate(chars))


start = numpy.random.randint(0, len(x_data) - 1)
pattern = x_data[start]
print("Random Seed characters:")
print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")

for i in range(50):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(vocab_len)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = num_to_char[index]
    if result == ' ': print("it's a blank space")
    
    seq_in = [num_to_char[value] for value in pattern]

    sys.stdout.write(result)

    pattern.append(index)
    pattern = pattern[1:len(pattern)]
    numpy.savetxt(os.getcwd() + "\\test.csv", prediction, delimiter=',')  #FAIL = seq_in, result, pattern
                                        # half right = prediction
