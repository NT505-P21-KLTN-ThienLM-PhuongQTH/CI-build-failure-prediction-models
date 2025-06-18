FROM python:3.9-alpine

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY consumer.simulation.py .
COPY requirements.simulation.txt .

RUN pip install --no-cache-dir -r requirements.simulation.txt

CMD ["python", "consumer.simulation.py"]