### To excecute the script

1. First put them in the same folder as "dataset"
2. Check that all the needed packages like Keras are already installed, otherwise you can refer to "installation instructions" bellow

```
CI-build-failure-prediction
│── data/
│── models/
│── notebooks/
│── results/
│── scripts/
│   │── test.py
│── src/
│   │── data/
│   │   │── preprocess.py
│   │   │── dataset_loader.py
│   │── models/
│   │   │── lstm_model.py
│   │   │── train.py
│   │── optimization/
│   │   │── GA_runner.py
│   │   │── optimizer.py
│   │   │── solution.py
│   │── utils/
│   │   │── Utils.py
│── README.md
│── requirements.txt
```
```aiignore
dvc remote add -d myremote s3://dvc
dvc remote modify myremote endpointurl http://<YOUR_IP>:9000
dvc remote modify myremote access_key_id minio
dvc remote modify myremote secret_access_key minio123
```