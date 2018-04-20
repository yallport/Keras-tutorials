#
#   This program will classify reviews from IMDB based on sentiment, positive or
#   negative.  We will used the IMDB database that comes with Keras. 
#   This data has already preprocessed the reviews.  This preprocessing 
#   replaces the actual works with the encoding.  So the second most 
#   popular word is replaced by 2, third most popular by 3, etc.    

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.callbacks import EarlyStopping
from keras.datasets import imdb

#   Supress warning and informational messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#   Set parameters for data to use
NUM_WORDS = 6000        # the top most n frequent words to consider
SKIP_TOP = 2            # Skip the top most words that are likely (the, and, a)
MAX_REVIEW_LEN = 100    # Max number of words from a review.

#   Load pre-processed sentiment classified review data from IMDB Database
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = NUM_WORDS,
                                        skip_top=SKIP_TOP)
#   Print a sample
#  returns word index vector (ex. [2, 4, 2, 2, 33, 2804, ...]) and class (0 or 1) 
print("encoded word sequence:", x_train[3], "class:", y_train[3])  


#   Pad and truncate the review word sequences so they are all the same length
x_train = sequence.pad_sequences(x_train, maxlen = MAX_REVIEW_LEN)
x_test = sequence.pad_sequences(x_test, maxlen = MAX_REVIEW_LEN)
print('x_train.shape:', x_train.shape, 'x_test.shape:', x_test.shape)

#   The Model
model = Sequential()
model.add(Embedding(NUM_WORDS, 64 ))
model.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(1, activation='sigmoid'))

#   Compile
model.compile(loss='binary_crossentropy',  
            optimizer='adam',              
            metrics=['accuracy'])

#   Train
BATCH_SIZE = 24
EPOCHS = 5
cbk_early_stopping = EarlyStopping(monitor='val_acc', patience=2, mode='max')
model.fit(x_train, y_train, BATCH_SIZE, epochs=EPOCHS, 
            validation_data=(x_test, y_test), 
            callbacks=[cbk_early_stopping] )

score, acc = model.evaluate(x_test, y_test,
                            batch_size=BATCH_SIZE)
print('test score:', score, ' test accuracy:', acc)