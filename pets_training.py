import logging
import pickle

from datetime import datetime

from tensorflow import keras, get_logger

logger = get_logger()
logger.setLevel(logging.ERROR)

# Carrega o dataset pronto para o treinamento
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

# Transforma o formato da imagem de um array de imagens de duas dimensões
# para um array de uma dimensão
model.add(keras.layers.Flatten())

# Essa são camadas neurais densely connected, ou fully connected
model.add(keras.layers.Dense(64))
model.add(keras.layers.Activation("relu"))

model.add(keras.layers.Dense(37))
model.add(keras.layers.Activation("softmax"))


# loss -> um valor escalar que tentamos minimizar durante o treinamento do modelo.
# Quanto menor o loss, mais próximas as previsões são das labels verdadeiras.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# batch_size -> Número de amostras processadas antes da atualização do modelo.
# epochs -> Número de passagens completas pelo conjunto de dados de treinamento
model.fit(x, y, batch_size=16, epochs=7)


scores = model.evaluate(x, y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

model.save(
    f"{scores[1]*100}{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_model.h5")
