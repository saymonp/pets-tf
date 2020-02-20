import os
import pickle
import random
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np


class DataManager(object):

    def __init__(self, IMG_SIZE: int, DATADIR: str, CATEGORIES: List[str]):
        self.IMG_SIZE = IMG_SIZE
        self.DATADIR = DATADIR
        self.CATEGORIES = CATEGORIES

    def test(self):
        """
        Mostra a imagem normalizada

        Torna a imagem em preto e branco
        Redimensiona o tamanho da imagem
        """
        img_array = []

        for category in self.CATEGORIES:
            path = os.path.join(self.DATADIR, category)
            for img in os.listdir(path):
                img_array = cv2.imread(os.path.join(
                    path, img), cv2.IMREAD_GRAYSCALE)
                break
            break

        new_array = cv2.resize(img_array, (self.IMG_SIZE, self.IMG_SIZE))
        plt.imshow(new_array, cmap="gray")
        plt.show()

    def normalize_data(self) -> List:
        training_data = []

        for category in self.CATEGORIES:
            print("Category " + category)
            path = os.path.join(self.DATADIR, category)
            class_num = self.CATEGORIES.index(category)

            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(
                        path, img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(
                        img_array, (self.IMG_SIZE, self.IMG_SIZE))
                    training_data.append([new_array, class_num])
                except Exception as e:
                    pass

        return training_data

    def create_training_data(self):
        """
        Gera os arquivos x.pickle e y.pickle prontos para o treinamento
        """
        training_data = self.normalize_data()

        # Embaralha as fotos
        random.shuffle(training_data)

        x = [] # Features
        y = [] # Labels

        for features, label in training_data:
            x.append(features)
            y.append(label)

        x = np.array(x).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)

        # Salva os dados prontos para o treinamento
        pickle_out = open("x.pickle", "wb")
        pickle.dump(x, pickle_out)
        pickle_out.close()

        pickle_out = open("y.pickle", "wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()


if __name__ == "__main__":

    IMG_SIZE = 80
    DATADIR = "images"
    CATEGORIES = ['Abyssian', 'American_bulldog', 'American_pit_bull', 'Basset_hound', 'Beagle', 'Bengal', 'Birdman',
                  'Bombay', 'Boxer', 'British_Shorthair',
                  'Chihuahua', 'Egyptian_Mau', 'English_Cocker_Spaniel', 'English_Setter', 'German_Shorthaired',
                  'Great_Pyrenees', 'Havanese', 'Japanese_Chin', 'Keeshond', 'Leonberger',
                  'Maine_Coon', 'Miniature_Pinscher', 'Newfoundland', 'Persian', 'Pomeranian', 'Pug', 'Ragdoll',
                  'Russian_Blue', 'Saint_Bernard', 'Samoyed',
                  'Scottish_Terrier', 'Shiba_Inu', 'Siamese', 'Sphynx', 'Staffordshire_Bull_Terrier',
                  'Wheaten_Terrier', 'Yorkshire_Terrier']

    data_manager = DataManager(IMG_SIZE, DATADIR, CATEGORIES)

    data_manager.create_training_data()
    # data_manager.test()