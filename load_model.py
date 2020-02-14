from tensorflow.keras.models import load_model
from tensorflow import get_logger

import logging
import pickle
import numpy as np

logger = get_logger()
logger.setLevel(logging.CRITICAL)

CATEGORIES = ['Abyssian', 'American_bulldog', 'American_pit_bull', 'Basset_hound', 'Beagle', 'Bengal', 'Birdman', 'Bombay', 'Boxer', 'British_Shorthair', 'Chihuahua', 'Egyptian_Mau', 'English_Cocker_Spaniel', 'English_Setter', 'German_Shorthaired', 'Great_Pyrenees', 'Havanese', 'Japanese_Chin',
              'Keeshond', 'Leonberger', 'Maine_Coon', 'Miniature_Pinscher', 'Newfoundland', 'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian_Blue', 'Saint_Bernard', 'Samoyed', 'Scottish_Terrier', 'Shiba_Inu', 'Siamese', 'Sphynx', 'Staffordshire_Bull_Terrier', 'Wheaten_Terrier', 'Yorkshire_Terrier']

# Carrega o modelo
model = load_model('98.1581807136535620200213231827_model.h5')
model.summary()

# Pega a label
label = pickle.load(open("y.pickle", "rb"))[1]
# Pega uma imagem do dataset normalizado
img = pickle.load(open("x.pickle", "rb"))[1]

# Modelos so keras são otimizados para fazer predições em um batch, ou coleções
# Adiciona a imagem em um batch que possui um só membro.
img = (np.expand_dims(img, 0))

# Faz a predição
predictions_single = model.predict(img)
# Predição da única imagem no batch:
predicao = np.argmax(predictions_single[0])

print(f"Predição: {predicao} -> {CATEGORIES[int(predicao)]}")

print(f"label: {label} -> {CATEGORIES[label]}")
