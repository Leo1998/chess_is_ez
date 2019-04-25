
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import time

from keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Activation, LeakyReLU
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

import numpy as np

def load_dataset(xfilepath, yfilepath):
    X = np.load(xfilepath, mmap_mode='r')
    Y = np.load(yfilepath, mmap_mode='r')
    
    print("Loaded dataset", X.shape, Y.shape)
    return (X, Y)

def create_model():
    model = Sequential()

    lReLUAlpha = 0.2

    model.add(Conv2D(5, kernel_size=(3,3), use_bias=False, data_format='channels_first', padding='same', input_shape=(5,8,8)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=lReLUAlpha))
    model.add(Conv2D(16, kernel_size=(3,3), use_bias=False, data_format='channels_first', padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=lReLUAlpha))
    model.add(Conv2D(16, kernel_size=(3,3), use_bias=False, data_format='channels_first', padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=lReLUAlpha))

    ###########################

    model.add(Conv2D(32, kernel_size=(3,3), use_bias=False, data_format='channels_first', padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=lReLUAlpha))
    model.add(Conv2D(32, kernel_size=(3,3), use_bias=False, data_format='channels_first', padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=lReLUAlpha))
    model.add(Conv2D(32, kernel_size=(3,3), use_bias=False, data_format='channels_first', padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=lReLUAlpha))

    ###########################

    model.add(Conv2D(64, kernel_size=(2,2), use_bias=False, data_format='channels_first', padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=lReLUAlpha))
    model.add(Conv2D(64, kernel_size=(2,2), use_bias=False, data_format='channels_first', padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=lReLUAlpha))
    model.add(Conv2D(64, kernel_size=(2,2), use_bias=False, data_format='channels_first', padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=lReLUAlpha))

    ###########################

    """model.add(Conv2D(128, kernel_size=(1,1), use_bias=False, data_format='channels_first', padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=lReLUAlpha))
    model.add(Conv2D(128, kernel_size=(1,1), use_bias=False, data_format='channels_first', padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=lReLUAlpha))
    model.add(Conv2D(128, kernel_size=(1,1), use_bias=False, data_format='channels_first', padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=lReLUAlpha))"""

    model.add(Flatten())

    model.add(Dense(1, activation='tanh'))

    model.compile(loss="mean_squared_error", optimizer='adam')

    #print model structure
    model.summary()
    return model

if __name__ == "__main__":
    dataset = load_dataset("parsed_data/X_10M.npy", "parsed_data/Y_10M.npy")

    model = create_model()

    #use this for retraining
    #model = load_model('models/')

    filepath="checkpoints/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False)

    try:
          model.fit(dataset[0], dataset[1],
                batch_size=512,
                epochs=8,
                shuffle=True,
                verbose=1,
                callbacks=[checkpoint],
                validation_split=0.0)
    except KeyboardInterrupt:
          model.save("models/interrupted-{}.model".format(time.time()))

    model.save("models/small-10M-8E.model")