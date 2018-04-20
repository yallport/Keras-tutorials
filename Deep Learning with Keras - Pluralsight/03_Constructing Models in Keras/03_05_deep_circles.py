import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import EarlyStopping

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# Helper functions
def plot_data(pl, X, y):

    pl.plot(X[y==0, 0], X[y==0, 1], 'ob', alpha=0.5)

    pl.plot(X[y==1, 0], X[y==1, 1], 'xr', alpha=0.5)

    pl.legend(['0', '1'])

    return pl


def plot_decision_boundary(model, X, y):

    amin, bmin = X.min(axis=0) - 0.1
    amax, bmax = X.max(axis=0) + 0.1

    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)

    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]

    c = model.predict(ab)

    Z = c.reshape(aa.shape)

    plt.figure(figsize=(12, 8))

    plt.contourf(aa, bb, Z, cmap='bwr', alpha=0.2)

    plot_data(plt, X, y)

    return plt


X, y = make_circles(n_samples=1000, factor=.6, noise=0.1, random_state=42)

pl = plot_data(plt, X, y)
pl.show()


# Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Implementing Layers and Neurons
model = Sequential()

model.add(Dense(4, input_shape=(2, ), activation='tanh', name='Hidden-1'))
model.add(Dense(4, activation='tanh', name='Hidden-2'))
model.add(Dense(1, activation='sigmoid', name='output_layer'))

model.summary()


model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])


# Plot model isn't working :(
# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


# Defining early stopping callback
early_callback = [EarlyStopping(monitor='val_acc', patience=5, mode='max')]

model.fit(X_train, y_train, epochs=100, verbose=1, callbacks=early_callback, validation_data=(X_test, y_test))


# Loss and Accuracy
eval_result = model.evaluate(X_test, y_test)
print("\n\nTest Loss: ", eval_result[0], "\nTest Accuracy: ", eval_result[1])


# Plotting the Decision Boundary
plot_decision_boundary(model, X, y).show()
