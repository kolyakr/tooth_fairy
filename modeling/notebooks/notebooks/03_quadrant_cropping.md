```python
import os
import sys

sys.path.append(os.path.abspath("../"))

import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random
from ultralytics import YOLO
from IPython.display import Image, display
from pathlib import Path
import pandas as pd
import torch
import cv2
import os
import numpy as np
from utils import segment_quadrants
```


```python
def draw_predicted_quadrants(result, see_boxes=True, see_labels=True):
    annotated_image = result.plot(boxes=see_boxes, labels=see_labels)
    return annotated_image
```


```python
def draw_ground_truth_teeth(base_img, label_path):
    lbl_path = Path(label_path)
    if not lbl_path.exists():
        print(f"Warning: No label file found at {lbl_path}")
        return base_img
        
    img_h, img_w = base_img.shape[:2]
    overlay = base_img.copy()
    
    base_quad_colors = [
        np.array([0, 255, 0]),   # Quadrant 0 (Upper Right)
        np.array([0, 165, 255]), # Quadrant 1 (Upper Left)
        np.array([255, 0, 0]),   # Quadrant 2 (Lower Left)
        np.array([200, 0, 200])  # Quadrant 3 (Lower Right)
    ]
    
    with open(lbl_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7: 
            continue
            
        cls_id = int(parts[0])
        
        quad_idx = cls_id // 8   
        tooth_idx = cls_id % 8   
        
        brightness_scale = 1.0 - (tooth_idx * 0.08) 
        
        color = (base_quad_colors[quad_idx] * brightness_scale).astype(int).tolist()
        
        coords = np.array([float(p) for p in parts[1:]]).reshape(-1, 2)
        coords[:, 0] *= img_w
        coords[:, 1] *= img_h
        poly_pts = np.int32(coords)
        
        cv2.fillPoly(overlay, [poly_pts], color)
        cv2.polylines(base_img, [poly_pts], isClosed=True, color=color, thickness=2)
        
        # ==========================================
        # THE DYNAMIC TEXT PLACEMENT FIX
        # ==========================================
        # Classes 0-15 are Upper Jaw (Quadrants 0 and 1)
        if cls_id < 16: 
            # Upper teeth: Find the highest point (min Y) and draw ABOVE it
            top_idx = np.argmin(coords[:, 1])
            text_x, text_y = int(coords[top_idx, 0]), int(coords[top_idx, 1])
            cv2.putText(
                base_img, f"T{cls_id}", (text_x, text_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
        # Classes 16-31 are Lower Jaw (Quadrants 2 and 3)
        else:
            # Lower teeth: Find the lowest point (max Y) and draw BELOW it
            bottom_idx = np.argmax(coords[:, 1])
            text_x, text_y = int(coords[bottom_idx, 0]), int(coords[bottom_idx, 1])
            cv2.putText(
                base_img, f"T{cls_id}", (text_x, text_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

    alpha = 0.4
    final_img = cv2.addWeighted(overlay, alpha, base_img, 1 - alpha, 0)
    
    return final_img
```


```python
def visualize_quadrants_and_teeth(result, label_path, title="Predicted Quadrants & GT Teeth"):
    quadrant_img = draw_predicted_quadrants(result, False, False)
    
    stacked_img = draw_ground_truth_teeth(quadrant_img, label_path)
    
    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(stacked_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.show()
```


```python
def plot_cropped_quadrants(result, margin=40, iterations=1):
    original_image = result.orig_img.copy()
    img_h, img_w = original_image.shape[:2] 
    
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    classes = result.boxes.cls.cpu().numpy().astype(int)
    polygons = result.masks.xy

    crops = []

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for i, (cls, box, polygon) in enumerate(zip(classes, boxes, polygons)):
        
        black_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        poly_points = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(black_mask, [poly_points], 255)
        
        kernel = np.ones((margin, margin), np.uint8)
        dilated_mask = cv2.dilate(black_mask, kernel, iterations=iterations)
        
        isolated_quadrant_before = cv2.bitwise_and(original_image, original_image, mask=black_mask)
        isolated_quadrant = cv2.bitwise_and(original_image, original_image, mask=dilated_mask)
        
        x1, y1, x2, y2 = box
        
        pad_x1 = max(0, x1 - margin)
        pad_y1 = max(0, y1 - margin)
        pad_x2 = min(img_w, x2 + margin)
        pad_y2 = min(img_h, y2 + margin)
        
        crop_before = isolated_quadrant_before[y1:y2, x1:x2]
        final_crop = isolated_quadrant[pad_y1:pad_y2, pad_x1:pad_x2]

        crops.append(final_crop)

        axes[0][i].imshow(cv2.cvtColor(crop_before, cv2.COLOR_BGR2RGB))
        axes[0][i].axis('off')
        axes[0][i].set_title(f"Before Margin:Quadrant {cls} Crop", fontsize=16)
        
        axes[1][i].imshow(cv2.cvtColor(final_crop, cv2.COLOR_BGR2RGB))
        axes[1][i].axis('off')
        axes[1][i].set_title(f"After Margin:Quadrant {cls} Crop", fontsize=16)

    plt.tight_layout()
    plt.show()

    return crops
```


```python
def visualize_full_pipeline(result, label_path, margin=40, title="Full Image", leave_black_bg=True):
    visualize_quadrants_and_teeth(result, label_path, title=title)
    
    original_image = result.orig_img.copy()
    full_h, full_w = original_image.shape[:2]
    
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    classes = result.boxes.cls.cpu().numpy().astype(int)
    polygons = result.masks.xy
    
    base_quad_colors = [
        np.array([0, 255, 0]),   # Q0
        np.array([0, 165, 255]), # Q1
        np.array([255, 0, 0]),   # Q2
        np.array([200, 0, 200])  # Q3
    ]
    
    quad_teeth = {0: [], 1: [], 2: [], 3: []}
    
    if Path(label_path).exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7: continue
                
                cls_id = int(parts[0])
                quad_idx = cls_id // 8
                tooth_idx = cls_id % 8
                
                brightness = 1.0 - (tooth_idx * 0.08)
                color = (base_quad_colors[quad_idx] * brightness).astype(int).tolist()
                
                coords = np.array([float(p) for p in parts[1:]]).reshape(-1, 2)
                coords[:, 0] *= full_w
                coords[:, 1] *= full_h
                
                quad_teeth[quad_idx].append({'id': cls_id, 'color': color, 'coords': coords})
                
    num_crops = len(classes)
    if num_crops == 0:
        print("Model detected no quadrants.")
        return
        
    fig, axes = plt.subplots(1, num_crops, figsize=(6 * num_crops, 6))
    if num_crops == 1: axes = [axes] 
        
    for i, (cls, box, polygon) in enumerate(zip(classes, boxes, polygons)):

        original_image = result.orig_img.copy()
        
        # We NO LONGER stretch x1, y1, x2, y2. We trust YOLO's bounding box.
        x1, y1, x2, y2 = box

        # 1. Create the Stage 1 Quadrant Mask
        black_mask = np.zeros((full_h, full_w), dtype=np.uint8)
        poly_points = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(black_mask, [poly_points], 255)
        
        kernel = np.ones((margin, margin), np.uint8)
        dilated_mask = cv2.dilate(black_mask, kernel, iterations=1)
        
        # 2. THE CLIPPING MISSION: Cut off the bad GT labels!
        clipped_teeth = []
        teeth_in_quad = quad_teeth.get(cls, [])
        
        for tooth in teeth_in_quad:
            pts = np.int32(tooth['coords'])
            
            # A. Draw the human-annotated tooth on a blank canvas
            tooth_mask = np.zeros((full_h, full_w), dtype=np.uint8)
            cv2.fillPoly(tooth_mask, [pts], 255)
            
            # B. Slice it! Keep ONLY the pixels that are inside the YOLO quadrant mask
            clipped_tooth_mask = cv2.bitwise_and(tooth_mask, dilated_mask)
            
            # C. Calculate the new math coordinates of the shortened tooth
            contours, _ = cv2.findContours(clipped_tooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest piece (in case the clipping split the root into tiny dots)
                largest_contour = max(contours, key=cv2.contourArea)
                new_coords = largest_contour.squeeze()
                
                # Ensure the new shape is still a valid polygon (at least 3 points)
                if new_coords.ndim == 2 and len(new_coords) >= 3:
                    tooth['coords'] = new_coords
                    clipped_teeth.append(tooth)
                    
        # Replace the bad old teeth with our new, cleanly clipped teeth
        teeth_in_quad = clipped_teeth

        # 3. Apply the margin padding to the original YOLO box
        pad_x1 = max(0, x1 - margin)
        pad_y1 = max(0, y1 - margin)
        pad_x2 = min(full_w, x2 + margin)
        pad_y2 = min(full_h, y2 + margin)

        if leave_black_bg:
            isolated_quadrant = cv2.bitwise_and(original_image, original_image, mask=dilated_mask)
            final_crop = isolated_quadrant[pad_y1:pad_y2, pad_x1:pad_x2]
        else:
            final_crop = original_image[pad_y1:pad_y2, pad_x1:pad_x2]

        crop_overlay = final_crop.copy()
        
        # 4. Draw the newly shifted, newly clipped teeth!
        for tooth in teeth_in_quad:
            shifted_coords = tooth['coords'].copy()
            shifted_coords[:, 0] -= pad_x1
            shifted_coords[:, 1] -= pad_y1
            poly_pts = np.int32(shifted_coords)
            
            cv2.fillPoly(crop_overlay, [poly_pts], tooth['color'])
            cv2.polylines(final_crop, [poly_pts], isClosed=True, color=tooth['color'], thickness=2)
            
            if tooth['id'] < 16: 
                top_idx = np.argmin(shifted_coords[:, 1])
                text_x, text_y = int(shifted_coords[top_idx, 0]), int(shifted_coords[top_idx, 1])
                cv2.putText(final_crop, f"T{tooth['id']}", (text_x, text_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, tooth['color'], 2)
            else:
                bottom_idx = np.argmax(shifted_coords[:, 1])
                text_x, text_y = int(shifted_coords[bottom_idx, 0]), int(shifted_coords[bottom_idx, 1])
                cv2.putText(final_crop, f"T{tooth['id']}", (text_x, text_y + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, tooth['color'], 2)
        
        alpha = 0.4
        final_crop = cv2.addWeighted(crop_overlay, alpha, final_crop, 1 - alpha, 0)
        
        bg_status = "Masked" if leave_black_bg else "Context"
        axes[i].imshow(cv2.cvtColor(final_crop, cv2.COLOR_BGR2RGB))
        axes[i].axis('off')
        axes[i].set_title(f"Q{cls} | {bg_status}\n(Clipped + Shifted)", fontsize=16)
        
    plt.tight_layout()
    plt.show()
```


```python
image_location = Path("../data/temp/train/images/train_629.png")
label_location = Path("../data/temp/train/labels/train_629.txt")

result = segment_quadrants(image_location)
```


```python
visualize_full_pipeline(result, label_location, margin=90, title=image_location.name)
```


    
![png](notebooks/03_quadrant_cropping_files/notebooks/03_quadrant_cropping_7_0.png)
    



    
![png](notebooks/03_quadrant_cropping_files/notebooks/03_quadrant_cropping_7_1.png)
    



```python
image_folder_path = Path("../data/temp/train/images")

all_images = list(image_folder_path.glob("*.png"))

all_images.sort(key=lambda x: int(x.name.split(".")[0].split("_")[1]))

len(all_images)
```




    634




```python
labels_folder_path = Path("../data/temp/train/labels")

all_labels = list(labels_folder_path.glob("*.txt"))

all_labels.sort(key=lambda x: int(x.name.split(".")[0].split("_")[1]))

len(all_labels)
```




    634




```python
for image_location, label_location in zip(all_images[:5], all_labels[:5]):
    result = segment_quadrants(str(image_location))
    visualize_full_pipeline(result, label_location, 90, image_location.name)
```


    
![png](notebooks/03_quadrant_cropping_files/notebooks/03_quadrant_cropping_10_0.png)
    



    
![png](notebooks/03_quadrant_cropping_files/notebooks/03_quadrant_cropping_10_1.png)
    



    
![png](notebooks/03_quadrant_cropping_files/notebooks/03_quadrant_cropping_10_2.png)
    



    
![png](notebooks/03_quadrant_cropping_files/notebooks/03_quadrant_cropping_10_3.png)
    



    
![png](notebooks/03_quadrant_cropping_files/notebooks/03_quadrant_cropping_10_4.png)
    



    
![png](notebooks/03_quadrant_cropping_files/notebooks/03_quadrant_cropping_10_5.png)
    



    
![png](notebooks/03_quadrant_cropping_files/notebooks/03_quadrant_cropping_10_6.png)
    



    
![png](notebooks/03_quadrant_cropping_files/notebooks/03_quadrant_cropping_10_7.png)
    



    
![png](notebooks/03_quadrant_cropping_files/notebooks/03_quadrant_cropping_10_8.png)
    



    
![png](notebooks/03_quadrant_cropping_files/notebooks/03_quadrant_cropping_10_9.png)
    



```python
for image_location, label_location in zip(all_images[:5], all_labels[:5]):
    result = segment_quadrants(str(image_location))
    visualize_full_pipeline(result, label_location, 40, image_location.name, leave_black_bg=False)
```


    
![png](notebooks/03_quadrant_cropping_files/notebooks/03_quadrant_cropping_11_0.png)
    



    
![png](notebooks/03_quadrant_cropping_files/notebooks/03_quadrant_cropping_11_1.png)
    



    
![png](notebooks/03_quadrant_cropping_files/notebooks/03_quadrant_cropping_11_2.png)
    



    
![png](notebooks/03_quadrant_cropping_files/notebooks/03_quadrant_cropping_11_3.png)
    



    
![png](notebooks/03_quadrant_cropping_files/notebooks/03_quadrant_cropping_11_4.png)
    



    
![png](notebooks/03_quadrant_cropping_files/notebooks/03_quadrant_cropping_11_5.png)
    



    
![png](notebooks/03_quadrant_cropping_files/notebooks/03_quadrant_cropping_11_6.png)
    



    
![png](notebooks/03_quadrant_cropping_files/notebooks/03_quadrant_cropping_11_7.png)
    



    
![png](notebooks/03_quadrant_cropping_files/notebooks/03_quadrant_cropping_11_8.png)
    



    
![png](notebooks/03_quadrant_cropping_files/notebooks/03_quadrant_cropping_11_9.png)
    



```python
def create_yolo_dirs(base_dir):
    for split in ['train', 'val']:
        os.makedirs(Path(base_dir) / split / 'images', exist_ok=True)
        os.makedirs(Path(base_dir) / split / 'labels', exist_ok=True)
```


```python
MASKED_PATH = "../data/teeth segmentation masked"
CONTEXT_PATH = "../data/teeth segmentation context"

create_yolo_dirs(MASKED_PATH)
create_yolo_dirs(CONTEXT_PATH)
```


```python
def generate_stage2_crop(result, label_path, base_out_dir, split_name, img_name, margin=40, leave_black_bg=True):
    original_image = result.orig_img.copy()
    full_h, full_w = original_image.shape[:2]
    
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    classes = result.boxes.cls.cpu().numpy().astype(int)
    polygons = result.masks.xy
    
    quad_teeth = {0: [], 1: [], 2: [], 3: []}
    if Path(label_path).exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7: continue
                cls_id = int(parts[0])
                quad_idx = cls_id // 8
                
                coords = np.array([float(p) for p in parts[1:]]).reshape(-1, 2)
                coords[:, 0] *= full_w
                coords[:, 1] *= full_h
                quad_teeth[quad_idx].append({'id': cls_id, 'coords': coords})
                
    for cls, box, polygon in zip(classes, boxes, polygons):
        x1, y1, x2, y2 = box

        black_mask = np.zeros((full_h, full_w), dtype=np.uint8)
        poly_points = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(black_mask, [poly_points], 255)
        
        kernel = np.ones((margin, margin), np.uint8)
        dilated_mask = cv2.dilate(black_mask, kernel, iterations=1)
        
        clipped_teeth = []
        for tooth in quad_teeth.get(cls, []):
            pts = np.int32(tooth['coords'])
            tooth_mask = np.zeros((full_h, full_w), dtype=np.uint8)
            cv2.fillPoly(tooth_mask, [pts], 255)
            
            clipped_tooth_mask = cv2.bitwise_and(tooth_mask, dilated_mask)
            contours, _ = cv2.findContours(clipped_tooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                new_coords = largest_contour.squeeze()
                if new_coords.ndim == 2 and len(new_coords) >= 3:
                    tooth['coords'] = new_coords
                    clipped_teeth.append(tooth)

        pad_x1 = max(0, x1 - margin)
        pad_y1 = max(0, y1 - margin)
        pad_x2 = min(full_w, x2 + margin)
        pad_y2 = min(full_h, y2 + margin)

        if leave_black_bg:
            isolated_quadrant = cv2.bitwise_and(original_image, original_image, mask=dilated_mask)
            final_crop = isolated_quadrant[pad_y1:pad_y2, pad_x1:pad_x2]
        else:
            final_crop = original_image[pad_y1:pad_y2, pad_x1:pad_x2]

        crop_h, crop_w = final_crop.shape[:2]
        
        yolo_lines = []
        for tooth in clipped_teeth:
            shifted_coords = tooth['coords'].copy()
            
            # 1. SHIFT
            shifted_coords[:, 0] -= pad_x1
            shifted_coords[:, 1] -= pad_y1
            
            norm_x = shifted_coords[:, 0] / crop_w
            norm_y = shifted_coords[:, 1] / crop_h
            
            norm_x = np.clip(norm_x, 0.0, 1.0)
            norm_y = np.clip(norm_y, 0.0, 1.0)
            
            stage2_cls_id = tooth['id'] % 8
            
            flat_coords = []
            for nx, ny in zip(norm_x, norm_y):
                flat_coords.extend([f"{nx:.6f}", f"{ny:.6f}"])
            
            yolo_lines.append(f"{stage2_cls_id} " + " ".join(flat_coords))

        if yolo_lines:
            base_name = Path(img_name).stem
            save_name = f"{base_name}_Q{cls}"
            
            # Save Image
            img_out_path = Path(base_out_dir) / split_name / 'images' / f"{save_name}.png"
            cv2.imwrite(str(img_out_path), final_crop)
            
            # Save Label
            txt_out_path = Path(base_out_dir) / split_name / 'labels' / f"{save_name}.txt"
            with open(txt_out_path, 'w') as f:
                f.write("\n".join(yolo_lines))
```


```python
train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42
)

splits = {
    'train': list(zip(train_imgs, train_lbls)),
    'val': list(zip(val_imgs, val_lbls))
}

ignore_files = [70, 220, 150, 258, 628, 201, 265, 173]

for split_name, data in splits.items():
    
    for img_path, lbl_path in data:

        if int(img_path.name.split(".")[0].split("_")[1]) in ignore_files:
            continue
        else:
            result = segment_quadrants(str(img_path)) 
            
            generate_stage2_crop(
                result=result, 
                label_path=lbl_path, 
                base_out_dir=MASKED_PATH, 
                split_name=split_name, 
                img_name=img_path.name, 
                margin=90, 
                leave_black_bg=True
            )
            
            generate_stage2_crop(
                result=result, 
                label_path=lbl_path, 
                base_out_dir=CONTEXT_PATH, 
                split_name=split_name, 
                img_name=img_path.name, 
                margin=40, 
                leave_black_bg=False
            )

```
