import logging
import pickle

from tensorflow import keras, get_logger

logger = get_logger()
logger.setLevel(logging.ERROR)

x = pickle.load(open("x.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

x = x/255.0

model = keras.Sequential()

model.add(keras.layers.Conv2D(64, (3, 3), input_shape=x.shape[1:]))
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Conv2D(64, (3, 3)))
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(64))
model.add(keras.layers.Activation("relu"))

model.add(keras.layers.Dense(37))
model.add(keras.layers.Activation("softmax"))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x, y, batch_size=16, epochs=7)

scores = model.evaluate(x, y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
