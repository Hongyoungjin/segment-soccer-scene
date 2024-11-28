from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)

# Train the model using "Calibration" Dataset
results = model.train(data="SoccerNet-calibration.yaml", epochs=100, imgsz=640)

print(results)