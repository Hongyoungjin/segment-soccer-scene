import numpy as np
import json
import os
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from parmap import parmap
from functools import partial

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class_mapping = {
    "Goal left crossbar": 0,
    "Goal left crossbar": 1,
    "Goal left post left": 2,
    "Goal left post right": 3,
    "Goal right crossbar": 4,
    "Goal right crossbar": 5,
    "Goal right post left": 6,
    "Goal right post right": 7,
    "Small rect. left top": 8,
    "Small rect. left main": 9,
    "Small rect. left bottom": 10,
    "Small rect. right top": 11,
    "Small rect. right main": 12,
    "Small rect. right bottom": 13,
}

# Image Shape
H, W = 540, 960

dataset_dirs = ["datasets/calibration", "datasets/calibration-2023"]
dirs = []
for dataset_dir in dataset_dirs:
    train_dir = os.path.join(dataset_dir, "train")
    valid_dir = os.path.join(dataset_dir, "valid")
    test_dir = os.path.join(dataset_dir, "test")
    dirs.append(train_dir)
    dirs.append(valid_dir)
    dirs.append(test_dir)
    

# for dir_name in tqdm(dirs, desc=f"Converting goal and penalty lines to YOLO format"):

def convert(dir_name):
    
    files = set(os.listdir(dir_name))
    files = [f.split(".")[0] for f in files if f.endswith('.jpg')]

    for file in files:
        json_file = os.path.join(dir_name, file + '.json')
        image_file = os.path.join(dir_name, file + '.jpg')
        
        # Rescale the image (540 x 960 → 640 x 640)
        image = cv2.imread(image_file)
        H, W, C = image.shape
        if H != 640 or W != 640:
            rescaled_image = cv2.resize(image, dsize=(640,640), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(image_file, rescaled_image)
        
        # Convert annotations into YOLO style
        with open(json_file, "r") as f:
            data = json.load(f)
            
        yolo_annotations = []
        for label, points in data.items():
        
            # Ignore objects with a single xy-coordinate
            # Ignore datapoints without labels in interest
            if len(points) > 1 and label in class_mapping.keys():
                
                class_id = class_mapping[label]
            
                annotated_points = []
                coordinates = []
                # YOLO Segmentation dataset requires at least 3 xy-coordinates, but SoccerNet only has 2 coordinates.
                for point in points:
                    coordinates.append([point['x'], point['y']])
                    
                coordinates = np.asarray(coordinates)
                # All coordinates should be between 0 and 1 (remove the normalization errors)
                coordinates = np.clip(coordinates, 0, 1)
                
                
                mid_coord = coordinates.mean(0)
                coordinates = np.vstack((coordinates, mid_coord))
                coordinates = np.sort(coordinates, axis=0)
                
                if len(coordinates) <=2:
                    print(f"Error in {file}")
                
                for coordinate in coordinates:
                    annotated_points.append(f"{coordinate[0]:.6f} {coordinate[0]:.6f}")
                    
                annotation_line = f"{class_id}" + " " + " ".join(annotated_points)
                yolo_annotations.append(annotation_line)
                
        with open(os.path.join(dir_name, file + '.txt'), "w") as f:
            f.write("\n".join(yolo_annotations))
        
        
def check_imagesize(dir_name):
    files = set(os.listdir(dir_name))
    files = [f.split(".")[0] for f in files if f.endswith('.jpg')]

    for file in files:
        image_file = os.path.join(dir_name, file + '.jpg')
        
        # Rescale the image (540 x 960 → 640 x 640)
        image = cv2.imread(image_file)
        H, W, C = image.shape
        
        if H != 640 or  W != 640:
            print(f"Error in {file}")
            
def count_annotation_points(dir_name):
    files = set(os.listdir(dir_name))
    files = [f.split(".")[0] for f in files if f.endswith('.jpg')]

    for file in files:
        annotation_file = os.path.join(dir_name, file + '.json')
        with open(annotation_file, 'r') as f:
            data = json.load(f)
            
        for label, points in data.items():
        
            if label in class_mapping.keys():
                if len(points) == 1:
                    print(f"Single Point Alert: Dataset: {dir_name}, Name: {file}")
                # if len(points) == 1:
                #     print(f"Single Point in {file}")
                
                
            
            
        
parmap.map(convert, dirs, pm_processes=16)
# parmap.map(check_imagesize, dirs, pm_processes=16)
# parmap.map(count_annotation_points, dirs, pm_processes=16)

    
        
    
    
    

