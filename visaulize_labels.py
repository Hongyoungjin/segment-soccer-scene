from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2


idx = '01245'
image = cv2.imread(f"datasets/calibration-2023/test/{idx}.jpg")
image = cv2.resize(image, dsize=(640,640), interpolation=cv2.INTER_CUBIC)

with open(f"datasets/calibration-2023/test/{idx}.json", "r") as f:
    data = json.load(f)
    
image = np.asarray(image)
H, W, C  = image.shape
ref_size = 640

fig = plt.figure()

plt.imshow(image)

for label, points in data.items():
    
    Xs, Ys = [], []
    for point in points:
        x_abs = point['x'] * ref_size
        y_abs = point['y'] * ref_size
        
        Xs.append(x_abs)
        Ys.append(y_abs)
        
    plt.plot(Xs, Ys, label=label)
    
plt.legend(fontsize='xx-small')
    
plt.savefig(f"{idx}.jpg")
    
    