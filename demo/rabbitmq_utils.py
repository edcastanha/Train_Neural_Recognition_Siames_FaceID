import pika
# import numpy as np
import json


def send_to_queue(device):
    # Configurar as credenciais de autenticação
    credentials = pika.PlainCredentials("sippe", "ep4X1!br")

    # Estabelecer conexão com o RabbitMQ
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host="localhost",
            credentials=credentials
        )
    )
    channel = connection.channel()

    # Declarar a fila
    channel.queue_declare(queue="detection_face", durable=True)

    # Enviar dados para a fila
    message = {
        "device": device
    }

    message_json = json.dumps(message)
    
    channel.basic_publish(exchange="sippe", routing_key="detection_face", body=message_json)

    # Fechar a conexão
    connection.close()

camera = "SIPPE1"
send_to_queue(camera)
