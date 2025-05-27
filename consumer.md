# Cài đặt các thư viện
!pip install pika python-dotenv dagshub

import pika
import os
import subprocess
import json
from dotenv import load_dotenv
from google.colab import drive
from dagshub.streaming import DagsHubFilesystem

# Mount Google Drive
drive.mount('/content/drive')
env_path = '/content/drive/MyDrive/ColabNotebooks/.env'
if os.path.exists(env_path):
    !cp -f {env_path} /content/.env
else:
    print("Warning: .env file not found in Google Drive.")

load_dotenv('/content/.env')

# Lấy thông tin từ .env
git_user = os.getenv("GITHUB_USER")
git_token = os.getenv("GITHUB_TOKEN")
rabbitmq_user = os.getenv("RABBITMQ_USER")
rabbitmq_password = os.getenv("RABBITMQ_PASSWORD")
rabbitmq_host = os.getenv("RABBITMQ_HOST")
rabbitmq_vhost = os.getenv("RABBITMQ_VHOST")
repo_url = os.getenv("DAGSHUB_REPO")
data_path = os.getenv("DAGSHUB_DATA_PATH")
rev = os.getenv("DAGSHUB_BRANCH")

# Connect to Dagshub
try:
    fs = DagsHubFilesystem(".", repo_url=repo_url, branch=rev)
except Exception as e:
    print("Error:", str(e))

# Pull repository
repo_dir = "/content/CI-build-failure-prediction-models"
if not os.path.exists(repo_dir):
    !git clone https://{git_user}:{git_token}@github.com/NT505-P21-KLTN-ThienLM-PhuongQTH/CI-build-failure-prediction-models.git {repo_dir}
%cd {repo_dir}
!git pull origin main

# Cài đặt requirements
try:
    subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
except subprocess.CalledProcessError as e:
    print(f"Error installing requirements: {e}")

# Copy .env vào thư mục source
env_destination = os.path.join(repo_dir, '.env')
if os.path.exists('/content/.env'):
    !cp /content/.env {env_destination}
else:
    print("Warning: .env file not found in /content")

# Kết nối tới RabbitMQ
credentials = pika.PlainCredentials(rabbitmq_user, rabbitmq_password)
parameters = pika.ConnectionParameters(
    host=rabbitmq_host,
    port=5672,
    virtual_host=rabbitmq_vhost,
    credentials=credentials
)
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

# Tạo hàng đợi bền bỉ
queue_name = 'training_queue'
channel.queue_declare(queue=queue_name, durable=True)
channel.queue_purge(queue=queue_name)

# Callback xử lý message
def callback(ch, method, properties, body):
    print("Received message...")
    try:
        # Phân tích message JSON để lấy model_type
        message = json.loads(body.decode())
        model_type = message.get("model_type", "lstm")  # Mặc định là 'lstm' nếu không có model_type
        if model_type not in ["lstm", "bilstm", "padding"]:
            raise ValueError(f"Invalid model_type: {model_type}. Must be 'lstm', 'bilstm', or 'padding'.")

        print(f"Running training container with model_type={model_type}...")
        # Chạy container Docker với tham số model_type
        docker_command = [
            "docker", "run", "--rm",
            "-v", f"{repo_dir}:/app",
            "ci-build-failure-prediction",
            "python", "pipeline.py", "--model_type", model_type
        ]
        process = subprocess.Popen(
            docker_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        for line in process.stdout:
            print(line, end="")  # In trực tiếp ra Colab
        process.wait()
        if process.returncode != 0:
            print(f"Training container failed with return code {process.returncode}")
        else:
            print("Training container completed successfully")
    except Exception as e:
        print(f"Error running training container: {str(e)}")
    ch.basic_ack(delivery_tag=method.delivery_tag)

# Tiêu thụ message
channel.basic_consume(queue=queue_name, on_message_callback=callback)
print(f"Waiting for training messages on queue '{queue_name}'...")
try:
    channel.start_consuming()
except KeyboardInterrupt:
    channel.stop_consuming()
finally:
    connection.close()