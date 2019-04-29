
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import time

from keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Activation, LeakyReLU
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam, Adadelta
from keras.callbacks import ModelCheckpoint

import numpy as np

def load_dataset(xfilepath, yfilepath):
    X = np.load(xfilepath, mmap_mode='r')
    Y = np.load(yfilepath, mmap_mode='r')
    
    print("Loaded dataset", X.shape, Y.shape)
    return (X, Y)

def create_model():
    model = Sequential()

    lReLUAlpha = 0.3

    model.add(Conv2D(5, kernel_size=(3,3), use_bias=False, data_format='channels_first', padding='same', input_shape=(5,8,8)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=lReLUAlpha))
    model.add(Conv2D(16, kernel_size=(3,3), use_bias=False, data_format='channels_first', padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=lReLUAlpha))
    model.add(Conv2D(16, kernel_size=(3,3), use_bias=False, data_format='channels_first', padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=lReLUAlpha))

    model.add(Conv2D(32, kernel_size=(3,3), use_bias=False, data_format='channels_first', padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=lReLUAlpha))
    model.add(Conv2D(32, kernel_size=(3,3), use_bias=False, data_format='channels_first', padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=lReLUAlpha))
    model.add(Conv2D(32, kernel_size=(3,3), use_bias=False, data_format='channels_first', strides=(2,2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=lReLUAlpha))

    model.add(Conv2D(64, kernel_size=(2,2), use_bias=False, data_format='channels_first', padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=lReLUAlpha))
    model.add(Conv2D(64, kernel_size=(2,2), use_bias=False, data_format='channels_first', padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=lReLUAlpha))
    model.add(Conv2D(64, kernel_size=(2,2), use_bias=False, data_format='channels_first', strides=(2,2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=lReLUAlpha))

    model.add(Conv2D(128, kernel_size=(1,1), use_bias=False, data_format='channels_first'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=lReLUAlpha))
    model.add(Conv2D(128, kernel_size=(1,1), use_bias=False, data_format='channels_first'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=lReLUAlpha))
    model.add(Conv2D(128, kernel_size=(1,1), use_bias=False, data_format='channels_first'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=lReLUAlpha))

    model.add(Flatten(data_format='channels_first'))

    model.add(Dense(1, activation='tanh'))

    opt=Adam(lr=0.001, decay=0.00001)
    #opt=Adadelta()
    model.compile(loss="mean_squared_error", optimizer=opt)

    #print model structure
    model.summary()
    return model

if __name__ == "__main__":
    dataset = load_dataset("parsed_data/X_1M.npy", "parsed_data/Y_1M.npy")

    model = create_model()

    #use this for retraining
    #model = load_model('models/net-1M-60E-v2.model')

    filepath="checkpoints/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False)

    try:
          model.fit(dataset[0], dataset[1],
                batch_size=256,
                epochs=10,
                shuffle=True,
                verbose=1,
                #callbacks=[checkpoint],
                validation_split=0.0)
    except KeyboardInterrupt:
          model.save("models/interrupted-{}.model".format(time.time()))

    model.save("models/test-1M-10E.model")