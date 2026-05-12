```python
import os
import sys

sys.path.append(os.path.abspath("../"))

from utils import segment_quadrants, apply_clahe
import json
from pathlib import Path
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import shutil
```


```python
ANN_PATH = Path("../data/_raw/Teeth Segmentation JSON/d2/ann")
IMAGES_PATH = Path("../data/_raw/Teeth Segmentation JSON/d2/img")

all_annotations = list(ANN_PATH.glob("*.json"))
all_images = list(IMAGES_PATH.glob("*.jpg"))

all_images.sort(key=lambda x: int(x.name.split(".")[0]))
all_annotations.sort(key=lambda x: int(x.name.split(".")[0]))

len(all_annotations), len(all_images)
```


```python
with open(all_annotations[0], "r", encoding="utf8") as f:
    data = json.load(f)

data.keys()
```


```python
data["size"]
```


```python
len(data["objects"])
```


```python
teeth = {}

for object in data["objects"]:
    cls = object["classTitle"]
    points = object["points"]["exterior"]

    teeth[int(cls)] = points
```


```python
teeth
```


```python
# IMAGES_PATH = Path("../data/_raw/dentex_data/training_data/unlabelled/xrays")
# IMAGES_PATH = Path("../data/_raw/Teeth Segmentation JSON/d2/img")
IMAGES_PATH = Path("../data/_raw/teeth_classification/YOLO/valid/images")


all_images = list(IMAGES_PATH.glob("*.jpg"))

```


```python
def plot_dataset_results(image_paths, conf=0.25, alpha=0.4):
    # Mapping colors to specific quadrants (RGB for Matplotlib)
    quad_colors = [
        (0, 255, 0),    # Green (UR)
        (255, 165, 0),  # Orange (UL)
        (0, 150, 255),  # Sky Blue (LL)
        (255, 0, 255)   # Magenta (LR)
    ]

    bad_images = []

    for img_path in image_paths:
        # --- 1. ENHANCE IMAGE FIRST ---
        # Apply your CLAHE function to get the high-contrast grayscale image
        enhanced_gray = apply_clahe(img_path)
        
        # Convert grayscale (1 channel) to BGR (3 channels) 
        # This is CRITICAL so we can draw COLOR masks on it.
        enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

        # --- 2. INFERENCE ---
        # Pass the ENHANCED image to the model instead of just the path
        # Assuming segment_quadrants can accept a numpy array
        filtered_result = segment_quadrants(enhanced_bgr, conf=conf)

        bad_images.append(img_path)

        # --- 3. VISUALIZATION ---
        # Use the enhanced image as your base for the overlay
        original_img = enhanced_bgr.copy()
        overlay = original_img.copy()
        img_name = os.path.basename(img_path)
        
        if filtered_result.masks is not None:
            masks_xy = filtered_result.masks.xy
            classes = filtered_result.boxes.cls.cpu().numpy().astype(int)

            for mask, cls_idx in zip(masks_xy, classes):
                color = quad_colors[cls_idx] if cls_idx < len(quad_colors) else (255, 255, 255)
                poly = np.array(mask, dtype=np.int32)
                cv2.fillPoly(overlay, [poly], color)

        # Blend overlay with the ENHANCED background
        output = cv2.addWeighted(overlay, alpha, original_img, 1 - alpha, 0)

        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.title(f"PanoDent AI (CLAHE Enhanced) - Segmented: {img_name}", fontsize=14)
        plt.axis('off')
        plt.show()

    return bad_images
```


```python
bad_images = plot_dataset_results(all_images, conf=0.01)
```


```python

```
