from ultralytics import YOLO
import os
import numpy as np
# Find the last train session
train_list = os.listdir("runs/segment")
indices = [f.replace('train',"") for f in train_list]
indices = [int(f) for f in indices if f != ""]

if len(indices) == 0:
    model = YOLO("runs/segment/train/weights/best.pt")  

# Load the latest model
max_idx = np.max(np.asarray(indices))
model = YOLO(f"runs/segment/train{max_idx}/weights/best.pt")  

# Train the model using "Calibration-2023" Dataset
results = model.train(data="SoccerNet-calibration-2023.yaml", epochs=50, imgsz=640)

print(results)