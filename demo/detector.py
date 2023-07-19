import cv2
import tensorflow as tf
import numpy as np
import uuid

def face_detection(frame):
    # Carregar o modelo de detecção de faces
    # face_model = tf.keras.models.load_model("modelo_deteccao_faces.h5")

    # Realizar a detecção de faces no frame
    # Converta o frame para escala de cinza, se necessário
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Redimensione o frame para um tamanho adequado para o modelo
    resized_frame = cv2.resize(gray_frame, (224, 224))

    # Normalizar os valores dos pixels do frame
    normalized_frame = resized_frame / 255.0

    # Adapte o formato do frame para o formato esperado pelo modelo
    input_frame = normalized_frame.reshape((1, 224, 224, 1))

    # Realize a predição de detecção de faces usando o modelo
    predictions = face_model.predict(input_frame)

    # Processar as previsões e retornar as informações das faces detectadas
    faces = process_predictions(predictions)

    # Criar o diretório de destino se não existir
    destination_folder = 'application_data/input_image'
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Gerar um nome de arquivo único para a imagem
    image_name = str(uuid.uuid1()) + '.jpg'

    # Salvar a imagem no diretório de destino
    image_path = os.path.join(destination_folder, image_name)
    cv2.imwrite(image_path, frame)

    # Retornar a URL da imagem salva
    url_path = os.path.abspath(image_path)
    
    return url_path
