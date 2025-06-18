import os
import json
import logging
import pika
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_cloudamqp_connection():
    """Connect to RabbitMQ (CloudAMQP)"""
    credentials = pika.PlainCredentials(
        username=os.getenv("RABBITMQ_USER"),
        password=os.getenv("RABBITMQ_PASSWORD")
    )
    parameters = pika.ConnectionParameters(
        host=os.getenv("RABBITMQ_HOST"),
        port=int(os.getenv("RABBITMQ_PORT", "5672")),
        virtual_host=os.getenv("RABBITMQ_VHOST"),
        credentials=credentials
    )
    return pika.BlockingConnection(parameters)

def simulate_consume():
    """Simulate job consumption and print info only"""
    load_dotenv()

    connection = get_cloudamqp_connection()
    channel = connection.channel()
    channel.queue_declare(queue='training_queue', durable=True)

    def callback(ch, method, properties, body):
        try:
            job = json.loads(body)
            model_name = job.get("model_name", "unknown")
            tuner = job.get("tuner", "ga")

            logger.info(f"Received job: model = {model_name}, tuner = {tuner}")
            print(f"➡ Simulating training job for model {model_name} with tuner {tuner}...")

            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            logger.error(f"Error processing job: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='training_queue', on_message_callback=callback)

    logger.info("Simulated Consumer started, waiting for training jobs...")
    channel.start_consuming()

if __name__ == "__main__":
    simulate_consume()