import os
import random
import sys
import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
import argparse
from dotenv import load_dotenv
from src.data.processing import get_dataset
from src.model.common.model_factory import ModelFactory
import pika
import json

# dagshub.init(repo_owner='hoaiphuongdg26', repo_name='CI-build-failure-prediction-inference', mlflow=True)
# mlflow.set_tracking_uri(os.getenv("MLFLOW_S3_ENDPOINT_URL"))
model_names = ["Stacked-LSTM", "Stacked-BiLSTM", "ConvLSTM", "Padding"]

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

def get_cloudamqp_connection():
    """Kết nối đến CloudAMQP"""
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

def training(model_name="Stacked-LSTM", tuner="ga", datasets=None):
    set_seed(42)
    """
    Run the full MLOps pipeline: data loading, training, and validation.
    Args:
        model_name (str): 'Stacked-LSTM', 'Stacked-BiLSTM', 'ConvLSTM', or 'Padding' to specify the model type.
        tuner (str): Hyperparameter tuning method (e.g., 'ga').
        datasets (dict): Dictionary of datasets for training and validation.
    """
    if datasets is None:
        raise ValueError("No datasets provided for online validation.")

    model_name = ModelFactory.get_model_name(model_name)
    mlflow.set_experiment(model_name)
    if mlflow.active_run() is not None:
        mlflow.end_run()

    with mlflow.start_run(run_name=f"{model_name}"):
        if model_name == "Padding":
            # Training PaddingModule độc lập
            ModelFactory.train_padding_module(
                model_name=model_name,
                datasets=datasets,
                input_dim=None,
                time_step=40,
                epochs=20,
                batch_size=32,
                r2_threshold=0.7,
                max_iterations=5
            )
        else:
            print("Running Online Validation and Selecting Bellwether...")
            bellwether_dataset, all_datasets, bellwether_model_uri = ModelFactory.run_online_validation(
                model_name=model_name,
                tuner=tuner,
                datasets=datasets
            )

            print("\nRunning Cross-Project Validation with Selected Bellwether...")
            ModelFactory.run_cross_project_validation(
                model_name=model_name,
                bellwether_dataset=bellwether_dataset,
                all_datasets=all_datasets,
                bellwether_model_uri=bellwether_model_uri,
                tuner=tuner
            )

def produce_training_jobs():
    """Gửi các job đào tạo vào queue CloudAMQP"""
    connection = get_cloudamqp_connection()
    channel = connection.channel()
    channel.queue_declare(queue='training_queue', durable=True)

    for model_name in model_names:
        job = {"model_name": model_name, "tuner": "ga"}
        channel.basic_publish(
            exchange='',
            routing_key='training_queue',
            body=json.dumps(job),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        print(f"Sent training job for {model_name} to queue")
    connection.close()

def consume_training_jobs():
    """Lấy job từ queue và chạy training"""
    load_dotenv()
    datasets = get_dataset(
        repo_url=os.getenv("DAGSHUB_REPO"),
        data_path=os.getenv("DAGSHUB_DATA_PATH"),
        rev=os.getenv("DAGSHUB_BRANCH"),
        # file_list=["getsentry_sentry.csv"],
        dagshub_token=os.getenv("DAGSHUB_TOKEN"),
    )

    connection = get_cloudamqp_connection()
    channel = connection.channel()
    channel.queue_declare(queue='training_queue', durable=True)

    def callback(ch, method, properties, body):
        job = json.loads(body)
        model_name = job["model_name"]
        tuner = job["tuner"]
        print(f"Processing training job for {model_name}...")
        try:
            training(model_name=model_name, tuner=tuner, datasets=datasets)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    channel.basic_qos(prefetch_count=1)  # Xử lý từng job một
    channel.basic_consume(queue='training_queue', on_message_callback=callback)
    print("Worker started, waiting for training jobs...")
    channel.start_consuming()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CI build failure prediction pipeline.")
    parser.add_argument("--mode", type=str, default="produce", choices=["produce", "consume"],
                        help="Mode: 'produce' to send jobs to queue, 'consume' to process jobs from queue.")
    parser.add_argument("--model_name", type=str, default="Stacked-LSTM", choices=model_names,
                        help=f"Model type to use: {model_names}. Used only in direct training mode.")
    args = parser.parse_args()

    load_dotenv()
    if args.mode == "produce":
        produce_training_jobs()
    elif args.mode == "consume":
        consume_training_jobs()
    else:
        datasets = get_dataset(
            repo_url=os.getenv("DAGSHUB_REPO"),
            data_path=os.getenv("DAGSHUB_DATA_PATH"),
            rev=os.getenv("DAGSHUB_BRANCH"),
            # file_list=["getsentry_sentry.csv"],
            dagshub_token=os.getenv("DAGSHUB_TOKEN"),
        )
        training(model_name=args.model_name, datasets=datasets)