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

if not os.path.exists(POS_PATH):
    os.makedirs(POS_PATH)
if not os.path.exists(NEG_PATH):
    os.makedirs(NEG_PATH)
if not os.path.exists(ANC_PATH):
    os.makedirs(ANC_PATH)

#  Movendo todos file da path lfw para a pasta data/negative
# for directory in os.listdir('lfw'):
    # for file in os.listdir(os.path.join('lfw', directory)):
    #     if file.endswith('.jpg'):
    #         EX_PATH = os.path.join('lfw', directory, file)
    #         NEW_PATH = os.path.join(NEG_PATH, file)
    #         os.replace(EX_PATH, NEW_PATH)

# Capturar Imagens de Ancora
POS_PATH_userID = os.path.join('data', 'positive','1001')
if not os.path.exists(POS_PATH_userID):
    os.makedirs(POS_PATH_userID)

ANC_PATH_userID = os.path.join('data', 'anchor','1001')
if not os.path.exists(ANC_PATH_userID):
    os.makedirs(ANC_PATH_userID)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
    # Logica capturar img 250x250
        frame = frame[120:120+250, 160:160+250, :]

        cv2.imshow('Captura 250x250', frame)        
    # Coletando imagem Anchor
        if cv2.waitKey(1) & 0xFF == ord('a'):
            # Criando nome unico para imagem
            imgName = os.path.join(ANC_PATH_userID, '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(imgName, frame)

    #  Coletando imagem Positive
        if cv2.waitKey(1) & 0xFF == ord('p'):
            # Criando nome unico para imagem
            imgName = os.path.join(POS_PATH_userID, '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(imgName, frame)
            cv2.imshow('Captura Positiva 250x250', frame)

    # Exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # cv2.imwrite(os.path.join(ANC_PATH, 'anchor.jpg'), frame)
            break

    else:
        print('Não foi possível acessar a câmera')       
        break
cap.release()
cv2.destroyAllWindows()

