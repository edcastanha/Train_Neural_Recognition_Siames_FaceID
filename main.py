import os
import time
import cv2
from capturasTrain.RMQ_DetectionFace import send_to_queue
from capturasTrain.DetectorFace import face_detection

admin = 'admin'
password = 'ep4X1!br'
url = 'sippe3.ddns-intelbras.com.br'
port = 5543

def process_frame(image, device):
    # Detectar faces no frame
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_detection(img)

    # Enviar informações para a fila do RabbitMQ
    send_to_queue(faces, device)

def main():
    # Configurações da captura de vídeo RTSP
    rtsp_url = f"rtsp://{admin}:{password}@{url}:{port}/cam/realmonitor?channel=1&subtype=1"
    # print(rtsp_url)
    device = "Camera3"

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    if cap.isOpened():
        while True:
            # Capturar um frame do vídeo RTSP
            ret, frame = cap.read()

            if ret:
                print("Frame Main capturado:", frame)
                # Processar o frame para detecção de faces
                process_frame(frame, device)

            time.sleep(2)  # Aguardar um curto período antes de capturar o próximo frame

        cap.release()

if __name__ == "__main__":
    main()
