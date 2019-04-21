
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.layers import Conv2D, Flatten, Dense
from keras.models import Sequential

import numpy as np

def load_dataset(filepath):
    d = np.load(filepath)
    X = d['arr_0'].astype(float)
    Y = d['arr_1'].astype(float)
    print("Loaded dataset", X.shape, Y.shape)
    return (X, Y)

def create_model():
    model = Sequential()
    model.add(Conv2D(5, kernel_size=(3,3), data_format='channels_first', activation='relu', padding='same', input_shape=(5,8,8)))
    model.add(Conv2D(16, kernel_size=(3,3), data_format='channels_first', activation='relu', padding='same'))
    model.add(Conv2D(16, kernel_size=(3,3), data_format='channels_first', activation='relu', padding='same'))

    model.add(Conv2D(32, kernel_size=(3,3), data_format='channels_first', activation='relu', padding='same'))
    model.add(Conv2D(32, kernel_size=(3,3), data_format='channels_first', activation='relu', padding='same'))
    model.add(Conv2D(32, kernel_size=(3,3), data_format='channels_first', activation='relu', padding='same'))

    model.add(Conv2D(64, kernel_size=(2,2), data_format='channels_first', activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=(2,2), data_format='channels_first', activation='relu', padding='same'))
    model.add(Conv2D(64, kernel_size=(2,2), data_format='channels_first', activation='relu', padding='same'))

    model.add(Conv2D(128, kernel_size=(1,1), data_format='channels_first', activation='relu', padding='same'))
    model.add(Conv2D(128, kernel_size=(1,1), data_format='channels_first', activation='relu', padding='same'))
    model.add(Conv2D(128, kernel_size=(1,1), data_format='channels_first', activation='relu', padding='same'))

    model.add(Flatten())

    model.add(Dense(1, activation='tanh'))

    model.compile(loss="mean_squared_error", optimizer='adam')

    #model.summary()
    return model

if __name__ == "__main__":
    dataset = load_dataset("parsed_data/dataset_1M.npz")

    model = create_model()

    model.fit(dataset[0], dataset[1],
          batch_size=512,
          epochs=8,
          shuffle=True,
          verbose=1,
          validation_split=0.0)

    model.save("models/small-1M-8E.model")