import pika
import numpy as np
import json

class RMQ_DetectionFace:
    def __init__(self, host="localhost", username="sippe", password="ep4X1!br"):
        self.host = host
        self.username = username
        self.password = password

    def send_to_queue(self,faces):
        # Verificar se as credenciais foram fornecidas
        if self.username is not None and self.password is not None:
            credentials = pika.PlainCredentials(self.username, self.password)
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=self.host,
                    credentials=credentials
                )
            )
        else:
            connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.host))

        channel = connection.channel()

        if not channel.is_open:
            print("Conexão com o RabbitMQ não estabelecida")
            return
        else:
            # Declarar a fila
            channel.queue_declare(queue="detection_face", durable=True)

            # Converter o frame em uma matriz NumPy antes de enviá-lo
            # frame_data = frame.tobytes()

            # Enviar dados para a fila
            message = {
                "faces": faces,
            }

            message_json = json.dumps(message)

            channel.basic_publish(exchange="sippe", routing_key="detection_face", body=message_json)

            # Fechar a conexão
            connection.close()
