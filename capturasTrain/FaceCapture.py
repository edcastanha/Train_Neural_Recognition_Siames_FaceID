import cv2
import os
import random
import uuid

class FaceCapture:
    def __init__(self, ra):
        self.ra = ra
        self.pos_path = os.path.join('data', 'positive', str(self.ra))
        self.anc_path = os.path.join('data', 'anchor', str(self.ra))

        print(self.pos_path)

        self.create_directory(self.pos_path)
        self.create_directory(self.anc_path)

    def create_directory(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def capture_images(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                # Logic to capture 250x250 image
                frame = frame[120:120+250, 160:160+250, :]

                cv2.imshow('Captura 250x250', frame)

                # Coletando imagem Anchor
                if cv2.waitKey(1) & 0xFF == ord('a'):
                    img_name = os.path.join(self.anc_path, '{}.jpg'.format(uuid.uuid1()))
                    cv2.imwrite(img_name, frame)

                #  Coletando imagem Positive
                if cv2.waitKey(1) & 0xFF == ord('p'):
                    img_name = os.path.join(self.pos_path, '{}.jpg'.format(uuid.uuid1()))
                    cv2.imwrite(img_name, frame)
                    cv2.imshow('Captura Positiva 250x250', frame)

                # Exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                print('Não foi possível acessar a câmera')
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ra = 1001
    face_capture = FaceCapture(ra)
    face_capture.capture_images()
