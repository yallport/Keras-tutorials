#   Demonstrates multiple input and output using the Functional API

from keras.layers import Input, Embedding, LSTM, Dense, concatenate
from keras.models import Model
from keras.utils import plot_model

im = Input(shape=(100,200), name='input_main')    # Main Input
ls = LSTM(10, name='lstm_main')(im)
dl = Dense(5, activation='relu', name='dense_lstm')(ls)

ix = Input(shape=(5,), name='input_aux')             # Aux Input
cc = concatenate([dl, ix], name='d_lstm_and_aux_in')         
x = Dense(32, activation='relu', name='dm_out')(cc)    
mo = Dense(1, activation='sigmoid', name='main_out')(x)

ao = Dense(2, name='aux_out')(cc)

model = Model(inputs=[im, ix], 
                outputs=[mo, ao])

plot_model(model, to_file='model.png', show_shapes=False)