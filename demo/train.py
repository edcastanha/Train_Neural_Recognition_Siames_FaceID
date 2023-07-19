# Package                      Version
# ---------------------------- --------
# matplotlib                   3.7.2
# opencv-python                4.8.0.74
# tensorflow                   2.10.1
# tensorflow-gpu               2.10.1
import cv2
import numpy as np
import os
import random
import uuid
from matplotlib import pyplot as plt

# Importando dependencias de TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten

# Modelos Base para redes neurais convolucionais com TensorFlow:
# Model(input=[inputImage, inputVerification], output=[1,0])
# Input(shape=(100,100,3))

# Configurando uso de GPU e limites de memória
# https://www.tensorflow.org/guide/gpu
# https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
        # Habilita uso de memória da GPU
            tf.config.experimental.set_memory_growth(gpu, True)
        # Limita uso de memória da GPU
        #    tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        except RuntimeError as e:
            print(e)
else:
    print('Não foi localizado nenhuma GPU')

# Verificando se as pasta (Positivo, Negativo e Ancora ) existem, senão criaremos
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')


# Capturar Imagens de Ancora
POS_PATH_userID = os.path.join('data', 'positive','1001')
if not os.path.exists(POS_PATH_userID):
    os.makedirs(POS_PATH_userID)

ANC_PATH_userID = os.path.join('data', 'anchor','1001')
if not os.path.exists(ANC_PATH_userID):
    os.makedirs(ANC_PATH_userID)

class TrainFace():
    # Modelos Base para redes neurais convolucionais com TensorFlow:
    anchor = tf.data.Dataset.list_files(ANC_PATH_userID+'\*.jpg').take(300)
    positive = tf.data.Dataset.list_files(POS_PATH_userID+'\*.jpg').take(300)
    negative = tf.data.Dataset.list_files(NEG_PATH+'\*.jpg').take(300)

    # TEST: Imprimindo os caminhos das imagens
    # dir_test = negative.as_numpy_iterator()
    # for i in range(10):
    #     print(dir_test.next())

    def preprocess(file_path):
        byte_img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(byte_img)
        img = tf.image.resize(img, [100, 100])
        img = img / 255.0
        # print(img)
        return img

    # preprocess(NEG_PATH+'\Aaron_Eckhart_0001.jpg')

    # Criando DataSet
    # anchor_ds = anchor.map(preprocess)
    positives = tf.data.Dataset.zip((anchor , positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
    negatives = tf.data.Dataset.zip((anchor , negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
    data = positives.concatenate(negatives)

# TEST: Imprimindo DataSet ANCHOR & POSITIVE
    # samplos = data.as_numpy_iterator()

    # for i in range(10):
    #     print(samplos.next())

    def preprocess_twin(input_img, validation_img, label):
        return (preprocess(input_img), preprocess(validation_img), label)
    

# TEST Class PreProccess
dataset = TrainFace()
dataset.preprocess()
