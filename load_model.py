from tensorflow.keras.models import load_model

import pickle
import numpy as np

model = load_model('98.1581807136535620200213231827_model.h5')
model.summary()

img = pickle.load(open("x.pickle", "rb"))[1]
img = (np.expand_dims(img,0))

label = pickle.load(open("y.pickle", "rb"))[1]

predictions_single = model.predict(img)

print(np.argmax(predictions_single[0]))
print(f"label: {label}")

