import tensorflow as tf
import numpy as np
import uuid
import cv2 as cv
from keras_facenet import FaceNet


class DetectorFace:
    def __init__(self, embedder):
        self.embedder = FaceNet()
        #  testar o acesso a GPU
        print("GPU: ", tf.test.is_gpu_available())


    def face_detection(image): 
        # Load the cascade
        if image is None:
            return False
        else:
            detections = embedder.extract(image, threshold=0.95)
            print(detections)
            return "Encontrado Face"
        
            # If you have pre-cropped images, you can skip the
            # detection step.
            # embeddings = embedder.embeddings(images)



