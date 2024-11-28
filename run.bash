#!/bin/bash

# Download calibration dataset
python3 dataset.py

# Unzip dataset
unzip datasets/calibration/train.zip -d datasets/calibration
unzip datasets/calibration/valid.zip -d datasets/calibration
unzip datasets/calibration/test.zip -d datasets/calibration
unzip datasets/calibration-2023/train.zip -d datasets/calibration-2023
unzip datasets/calibration-2023/valid.zip -d datasets/calibration-2023
unzip datasets/calibration-2023/test.zip -d datasets/calibration-2023

# Convert SoccenNet Dataset to YOLO-seg Dataset format
python3 SoccerNet2YOLO.py

# Train pre-trained model using "Calibration" dataset
python3 train_calibration.py

# Additionally train using "Calibration-2023" dataset
python3 train_calibration-2023.py


