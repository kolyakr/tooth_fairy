```python
import os
import sys

sys.path.append(os.path.abspath("../"))

import re
import json
from pathlib import Path
from utils import segment_quadrants
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.gridspec as gridspec
import shutil
from collections import defaultdict
import random
```


```python
TRAIN_IMAGES_DIST = Path("../data/teeth_detection/train/images")
TRAIN_LABELS_DIST = Path("../data/teeth_detection/train/labels")
```


```python
TRAIN_IMAGES_SRC = Path("../data/_raw/teeth detection/d2/img")
TRAIN_LABELS_SRC = Path("../data/_raw/teeth detection/d2/ann")

all_images = list(TRAIN_IMAGES_SRC.glob("*.jpg"))
all_labels = list(TRAIN_LABELS_SRC.glob("*.json"))

all_images.sort(key=lambda x: int(x.name.split(".")[0]))
all_labels.sort(key=lambda x: int(x.name.split(".")[0]))

all_images[:2], all_labels[:2]
```


```python
processed_labels = []

for label_path in all_labels:
    with open(label_path, "r", encoding="utf8") as f:
        data = json.load(f)

    image_teeth_bboxes = []

    for obj in data["objects"]:
        class_name = obj["classTitle"]
        points = obj["points"]["exterior"] 
        
        if not points:
            continue
            
        x_coords, y_coords = zip(*points)
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        w = x_max - x_min
        h = y_max - y_min
        
        image_teeth_bboxes.append({
            "cls": class_name,
            "bbox": np.array([int(x_min), int(y_min), int(w), int(h)], dtype=np.int32)
        })
    
    processed_labels.append(image_teeth_bboxes)

all_labels = processed_labels
```


```python
len(all_labels), len(all_images)
```


```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_full_diagnostic(image_path, labels_input, padding=50):
    if isinstance(labels_input, list):
        all_labels_dict = {str(ann['cls']): ann['bbox'] for ann in labels_input}
    else:
        all_labels_dict = labels_input

    result = segment_quadrants(image_path, conf=0.01)
    processed_image = result.orig_img.copy()
    h_img, w_img = processed_image.shape[:2]
    filename = getattr(image_path, 'name', str(image_path))
    
    if result.boxes is None or len(result.boxes) == 0:
        print(f"⚠️ No quadrants detected in {filename}")
        return

    boxes = result.boxes.xyxy.cpu().numpy().astype(np.int32)
    classes = result.boxes.cls.cpu().numpy().astype(np.int32)

    quad_colors = [(0, 255, 0), (0, 165, 255), (255, 150, 0), (255, 0, 255)]
    tooth_yellow = (0, 255, 255)
    
    available_keys = [str(i) for i in range(1, 33)]
    quad_map = {
        "0": [k for k in available_keys if 1 <= int(k) <= 8],   
        "1": [k for k in available_keys if 9 <= int(k) <= 16],  
        "2": [k for k in available_keys if 17 <= int(k) <= 24], 
        "3": [k for k in available_keys if 25 <= int(k) <= 32], 
    }

    fig = plt.figure(figsize=(16, 18))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 1, 1])
    
    overlay_pano = processed_image.copy()

    # Draw Tooth Boxes (Panorama)
    for t_id, bbox in all_labels_dict.items():
        x, y, w, h = bbox
        cv2.rectangle(overlay_pano, (x, y), (x + w, y + h), tooth_yellow, 2)

    # Draw Quadrant Boxes (Panorama)
    for box, cls_id in zip(boxes, classes):
        x1, y1, x2, y2 = box
        color = quad_colors[int(cls_id)] if int(cls_id) < 4 else (255, 255, 255)
        cv2.rectangle(overlay_pano, (x1, y1), (x2, y2), color, 3)
        # Reduced Scale: 1.5 -> 0.8 | Thickness: 4 -> 2
        cv2.putText(overlay_pano, f"Q{cls_id}", (x1 + 10, y1 + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(cv2.cvtColor(overlay_pano, cv2.COLOR_BGR2RGB))
    ax0.set_title(f"Diagnostic Analysis: {filename}", fontsize=18, fontweight='bold')
    ax0.axis('off')

    grid_pos = {0: (1, 0), 1: (1, 1), 2: (2, 0), 3: (2, 1)}

    for box, cls_id in zip(boxes, classes):
        if int(cls_id) not in grid_pos: continue 
        
        x1, y1, x2, y2 = box
        px1, py1 = max(0, x1 - padding), max(0, y1 - padding)
        px2, py2 = min(w_img, x2 + padding), min(h_img, y2 + padding)
        
        crop = processed_image[py1:py2, px1:px2].copy()
        quad_color = quad_colors[int(cls_id)]
        target_teeth = quad_map.get(str(cls_id), [])
        found_count = 0

        for t_id in target_teeth:
            if t_id in all_labels_dict:
                found_count += 1
                tx, ty, tw, th = all_labels_dict[t_id]
                lx, ly = tx - px1, ty - py1
                
                # Tooth BBox
                cv2.rectangle(crop, (lx, ly), (lx + tw, ly + th), quad_color, 2)
                
                # --- LABEL OPTIMIZATION ---
                label = f"{t_id}"  # Just the ID number for maximum space
                # Reduced Scale: 0.7 -> 0.45 | Thickness: 2 -> 1
                font_scale, thickness = 0.45, 1
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Smaller background tab
                cv2.rectangle(crop, (lx, ly - lh - 6), (lx + lw + 4, ly), quad_color, -1)
                cv2.putText(crop, label, (lx + 2, ly - 4), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        row, col = grid_pos[int(cls_id)]
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Quadrant {cls_id} | {found_count} Teeth", 
                     fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
```


```python
for label, image in zip(all_labels[:3], all_images[:3]):
    plot_full_diagnostic(image, label, padding=50)
```


```python
def prepare_stage2_data(image_path, all_labels_dict, padding=50):
    IMAGES_DIR = Path("../data/teeth_detection/train/images")
    LABELS_DIR = Path("../data/teeth_detection/train/labels")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    base_name = Path(image_path).stem
    
    # 1. STAGE 1: Get Quadrants
    result = segment_quadrants(image_path, conf=0.01)
    if result.boxes is None or len(result.boxes) == 0:
        return

    processed_image = result.orig_img 
    h_img, w_img = processed_image.shape[:2]

    boxes = result.boxes.xyxy.cpu().numpy().astype(np.int32)
    classes = result.boxes.cls.cpu().numpy().astype(np.int32)

    # FDI Mapping
    available_keys = [str(i) for i in range(1, 33)]
    quad_map = {
        "0": [k for k in available_keys if 1 <= int(k) <= 8],   
        "1": [k for k in available_keys if 9 <= int(k) <= 16],  
        "2": [k for k in available_keys if 17 <= int(k) <= 24], 
        "3": [k for k in available_keys if 25 <= int(k) <= 32], 
    }

    saved_count = 0

    for box, cls_id in zip(boxes, classes):
        q_idx = int(cls_id)
        
        # File naming logic
        existing_files = list(IMAGES_DIR.glob(f"{base_name}_q{q_idx}_idx*.jpg"))
        if existing_files:
            idx_list = []
            for f in existing_files:
                match = re.search(r'_idx(\d+)', f.stem)
                if match: idx_list.append(int(match.group(1)))
            new_idx = max(idx_list) + 1 if idx_list else 0
        else:
            new_idx = 0

        save_name = f"{base_name}_q{q_idx}_idx{new_idx}"
        img_filename = IMAGES_DIR / f"{save_name}.jpg"
        txt_filename = LABELS_DIR / f"{save_name}.txt"

        if img_filename.exists(): continue

        x1, y1, x2, y2 = box
        px1, py1 = max(0, x1 - padding), max(0, y1 - padding)
        px2, py2 = min(w_img, x2 + padding), min(h_img, y2 + padding)
        
        crop = processed_image[py1:py2, px1:px2]
        crop_h, crop_w = crop.shape[:2]

        yolo_labels = []
        target_teeth = quad_map.get(str(q_idx), [])

        for t_id in target_teeth:
            # --- THE FIX IS HERE ---
            # all_labels_dict is now a direct map: { "8": array([...]), ... }
            if t_id in all_labels_dict:
                t_num = int(t_id)
                
                # Mirrored 8-class mapping
                if q_idx == 0:   class_id = 8 - t_num
                elif q_idx == 1: class_id = t_num - 9
                elif q_idx == 2: class_id = 24 - t_num
                elif q_idx == 3: class_id = t_num - 25
                
                class_id = int(np.clip(class_id, 0, 7))
                
                # Get [x_min, y_min, width, height]
                tx, ty, tw, th = all_labels_dict[t_id]
                
                # Transform to local crop
                lx_tl = tx - px1
                ly_tl = ty - py1
                
                # YOLO format: xc, yc, w, h (normalized)
                norm_xc = (lx_tl + (tw / 2)) / crop_w
                norm_yc = (ly_tl + (th / 2)) / crop_h
                norm_w = tw / crop_w
                norm_h = th / crop_h
                
                yolo_labels.append(f"{class_id} {norm_xc:.6f} {norm_yc:.6f} {norm_w:.6f} {norm_h:.6f}")

        if yolo_labels:
            cv2.imwrite(str(img_filename), crop)
            with open(txt_filename, "w") as f:
                f.write("\n".join(yolo_labels))
            saved_count += 1

    if saved_count > 0:
        print(f"✅ Processed: {base_name} | Saved {saved_count} quadrants")
```


```python
not_include_idx = [135, 202, 212, 214, 215, 271, 293, 367, 393, 402, 595, 277, 533, 543, 311, 423, 452, 495]
```


```python
for label_dict, image_path in zip(all_labels, all_images):
    file_num = int(image_path.stem) 
    
    if file_num in not_include_idx:
        print(f"Skipping filename: {file_num}") 
        continue
    else:
        prepare_stage2_data(image_path, label_dict)
```


```python
TRAIN_IMAGES_SRC = Path("../data/_raw/segmentation/dentex_data/training_data/quadrant_enumeration/xrays")
TRAIN_LABELS_SRC = Path("../data/_raw/segmentation/dentex_data/training_data/quadrant_enumeration/train_quadrant_enumeration.json")

all_images = list(TRAIN_IMAGES_SRC.glob("*.png"))

all_images.sort(key=lambda x: int(x.name.split(".")[0].split("_")[1]))

all_images[:10]
```


```python
with open(TRAIN_LABELS_SRC, "r", encoding="utf8") as f:
    data = json.load(f)

data.keys()
```


```python
data["categories_1"], data["categories_2"]
```


```python
data["annotations"][0]
```


```python
ann = data["annotations"]

ann.sort(key=lambda x: x["image_id"])
```


```python
def convert_to_global_class(q, local):
    q = int(q)
    local = int(local)
    
    if q == 0:   
        return 8 - local
    elif q == 1: 
        return 9 + local
    elif q == 2: 
        return 24 - local
    elif q == 3: 
        return 25 + local
    return 0
```


```python
image_id_to_filename = {img['id']: img['file_name'] for img in data['images']}
filename_to_idx = {path.name: i for i, path in enumerate(all_images)}

all_labels = [{} for _ in range(len(all_images))]

for teeth in ann:
    fname = image_id_to_filename[teeth["image_id"]]
    
    if fname not in filename_to_idx:
        continue
        
    idx = filename_to_idx[fname]
    
    cls = convert_to_global_class(teeth["category_id_1"], teeth["category_id_2"])
    
    bbox = np.array(teeth["bbox"], dtype=np.int32)
    
    all_labels[idx][str(cls)] = bbox
```


```python
len(all_labels), len(all_images)
```


```python
for label, image in zip(all_labels[:3], all_images[:3]):
    plot_full_diagnostic(image, label, padding=50)
```


```python
not_include_idx = [
        24, 70, 133, 150, 265, 486, 507, 512, 527, 584, 592
]
```


```python
for label_dict, image_path in zip(all_labels, all_images):
    file_num = int(image_path.stem.split("_")[-1]) 
    
    if file_num in not_include_idx:
        print(f"Skipping filename: {file_num}") 
        continue
    else:
        prepare_stage2_data(image_path, label_dict)
```


```python
TRAIN_IMAGES_SRC = Path("../data/_raw/teeth detection/ufba-425/numbering_xrays")
TRAIN_LABELS_SRC = Path("../data/_raw/teeth detection/ufba-425/boundinng_boxes")

all_images = list(TRAIN_IMAGES_SRC.glob("*.jpg"))
all_labels = list(TRAIN_LABELS_SRC.glob("*.txt"))

all_images[:3], all_labels[:3]
```




    ([PosixPath('../data/_raw/teeth detection/ufba-425/numbering_xrays/cate2-00001_jpg.rf.c3b2aa9110036dcca66072a10dff749f.jpg'),
      PosixPath('../data/_raw/teeth detection/ufba-425/numbering_xrays/cate6-00021_jpg.rf.a5a374247826ed31cded9a43e2982f6e.jpg'),
      PosixPath('../data/_raw/teeth detection/ufba-425/numbering_xrays/cate1-00032_jpg.rf.ae32059cc3a14daeee304d986af6df35.jpg')],
     [PosixPath('../data/_raw/teeth detection/ufba-425/boundinng_boxes/cate4-00095_jpg.rf.da7ee68b3c7d4726f45ebb0f3a2ee4cf.txt'),
      PosixPath('../data/_raw/teeth detection/ufba-425/boundinng_boxes/cate7-00074_jpg.rf.12a01f3dc902c5e705afe61f89aa0f44.txt'),
      PosixPath('../data/_raw/teeth detection/ufba-425/boundinng_boxes/cate2-00035_jpg.rf.903bb19bbb06a11739abeb245ad6d4c9.txt')])




```python
from PIL import Image

final_images = []
final_labels = []

for img_path in all_images:
    label_path = TRAIN_LABELS_SRC / (img_path.stem + ".txt")
    
    if not label_path.exists():
        continue

    with Image.open(img_path) as img:
        img_w, img_h = img.size

    image_labels_dict = {}
    
    with open(label_path, "r", encoding="utf8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5: 
                continue
            
            cls_id = int(parts[0])
            xc, yc, w_n, h_n = map(float, parts[1:])
            
            tooth_id = str(cls_id + 1)
            
            w_px = w_n * img_w
            h_px = h_n * img_h
            
            x_tl = int((xc * img_w) - (w_px / 2))
            y_tl = int((yc * img_h) - (h_px / 2))
            
            image_labels_dict[tooth_id] = np.array([
                max(0, x_tl), 
                max(0, y_tl), 
                int(w_px), 
                int(h_px)
            ], dtype=np.int32)

    if image_labels_dict:
        final_images.append(img_path)
        final_labels.append(image_labels_dict)

all_images = final_images
all_labels = final_labels
```


```python
len(all_images), len(all_labels)
```




    (425, 425)




```python
for label, image in zip(all_labels[:3], all_images[:3]):
    plot_full_diagnostic(image, label, padding=57)
```


    
![png](notebooks/04_1_teeth_detection_dataset_files/notebooks/04_1_teeth_detection_dataset_24_0.png)
    



    
![png](notebooks/04_1_teeth_detection_dataset_files/notebooks/04_1_teeth_detection_dataset_24_1.png)
    



    
![png](notebooks/04_1_teeth_detection_dataset_files/notebooks/04_1_teeth_detection_dataset_24_2.png)
    



```python
not_include_idx = [
    "cate10-00060_jpg.rf.49a69a671bd365a7282c5a449ede5306.jpg",
    "cate6-00147_jpg.rf.a1ad961c2145387bf9233353abd43f72.jpg",
    "cate6-00065_jpg.rf.2c543260f3d2116dc86c6bac2da6b296.jpg",
    "cate6-00014_jpg.rf.7f16a15b6dfcdc9c6faaaba7a56a8f13.jpg",
    "cate2-00023_jpg.rf.6a182c232906217c5f14b12c0bd4be35.jpg",
    "cate9-00034_jpg.rf.83b481150a51a0b186a1f4bb0f1910a6.jpg",
    "cate6-00003_jpg.rf.a74ff38dfc403c36db0b4677b6499b09.jpg",
    "cate8-00029_jpg.rf.d7de8d30e40ec7f6102668693f1f3fd5.jpg",
    "cate6-00159_jpg.rf.d670e83aa91a1a86fe0596aa604db6bf4.jpg",
    "cate6-00099_jpg.rf.66421231c2c5a93538c461b06d72087.jpg",
    "cate6-00141_jpg.rf.2cccae1daf54f19feee6d6293d0a2d83.jpg",
    "cate8-00299_jpg.rf.039ed8c0fd89fa56ee421de35d30bd14.jpg",
    "cate4-00087_jpg.rf.cc485441335c2f1f9be018eb5133a886.jpg",
    "cate8-00030_jpg.rf.844bed08b47e4bf086281d9293b92ca3.jpg",
    "cate8-00431_jpg.rf.e9a753767ad65be54775db7af883152b.jpg",
    "cate6-00051_jpg.rf.117e23e7e710bf1772fd334b0b4b67b4.jpg",
    "cate10-00066_jpg.rf.c1d56f5d8a5857b0c54d55c5760fec64.jpg",
    "cate3-00009_jpg.rf.8863a6fcb029e6d6766933be7c236cf7.jpg",
    "cate5-00060_jpg.rf.aa377e2b8cdd8e7b62b2b2ca9b9b4085.jpg",
    "cate4-00111_jpg.rf.1d271e4bcca3671a387db2ec5f0d4341.jpg",
    "cate6-00151_jpg.rf.e35e483dfeba488d47f3f3fd704172e8.jpg",
    "cate10-00056_jpg.rf.e283da4454bc0607fd574ffbdbb20db6.jpg",
    "cate7-00004_jpg.rf.ca37998477725a8349b56db33dffa06e.jpg",
    "cate6-00093_jpg.rf.7ee6d6ffc4ba11e6e96146f65800cb82.jpg",
    "cate7-00002_jpg.rf.eaf80e16fb8ac22309f8365ee3511943.jpg"
]
```


```python
TRAIN_IMAGES_SRC / not_include_idx[0]
```




    PosixPath('../data/_raw/teeth detection/ufba-425/numbering_xrays/cate10-00060_jpg.rf.49a69a671bd365a7282c5a449ede5306.jpg')




```python
not_include_idx = [(TRAIN_IMAGES_SRC / path) for path in not_include_idx]

not_include_idx[:5]
```




    [PosixPath('../data/_raw/teeth detection/ufba-425/numbering_xrays/cate10-00060_jpg.rf.49a69a671bd365a7282c5a449ede5306.jpg'),
     PosixPath('../data/_raw/teeth detection/ufba-425/numbering_xrays/cate6-00147_jpg.rf.a1ad961c2145387bf9233353abd43f72.jpg'),
     PosixPath('../data/_raw/teeth detection/ufba-425/numbering_xrays/cate6-00065_jpg.rf.2c543260f3d2116dc86c6bac2da6b296.jpg'),
     PosixPath('../data/_raw/teeth detection/ufba-425/numbering_xrays/cate6-00014_jpg.rf.7f16a15b6dfcdc9c6faaaba7a56a8f13.jpg'),
     PosixPath('../data/_raw/teeth detection/ufba-425/numbering_xrays/cate2-00023_jpg.rf.6a182c232906217c5f14b12c0bd4be35.jpg')]




```python
for label_dict, image_path in zip(all_labels, all_images):
    if image_path in not_include_idx:
        print(f"Skipping filename: {file_num}") 
        continue
    else:
        prepare_stage2_data(image_path, label_dict)
```

    ✅ Processed: cate2-00001_jpg.rf.c3b2aa9110036dcca66072a10dff749f | Saved 4 quadrants
    ✅ Processed: cate6-00021_jpg.rf.a5a374247826ed31cded9a43e2982f6e | Saved 4 quadrants
    ✅ Processed: cate1-00032_jpg.rf.ae32059cc3a14daeee304d986af6df35 | Saved 4 quadrants
    ✅ Processed: cate5-00054_jpg.rf.e019e4777bd2fa0b1dafca12c3895abc | Saved 4 quadrants
    ✅ Processed: cate8-00113_jpg.rf.e711fdb40db6f1fd0af5ff676caa2e6c | Saved 4 quadrants
    ✅ Processed: cate8-00202_jpg.rf.482f78de446d3a2357221c74686d9709 | Saved 4 quadrants
    ✅ Processed: cate8-00179_jpg.rf.42c525b0e06616cafa26bd93fbaba770 | Saved 4 quadrants
    ✅ Processed: cate2-00127_jpg.rf.18c8c7a200ae4f5a3516d2049473274e | Saved 4 quadrants
    Skipping filename: 633
    ✅ Processed: cate10-00054_jpg.rf.539928d0af38aeef8526d002f545d108 | Saved 4 quadrants
    ✅ Processed: cate2-00010_jpg.rf.840b78b87aefc5c054c0a14ce7343e65 | Saved 4 quadrants
    ✅ Processed: cate8-00131_jpg.rf.9755ad7a2010d677500a57fe7291285f | Saved 4 quadrants
    ✅ Processed: cate7-00054_jpg.rf.bac31e472d7ed20b569607f8e06ca7c8 | Saved 4 quadrants
    ✅ Processed: cate7-00090_jpg.rf.4a5fbd54ee7ac07b6e065a9789c20223 | Saved 4 quadrants
    ✅ Processed: cate5-00109_jpg.rf.b228b13645b1197647b308d0c2250412 | Saved 2 quadrants
    Skipping filename: 633
    Skipping filename: 633
    ✅ Processed: cate8-00395_jpg.rf.ad7f94ddf29369b47defa5e501632079 | Saved 4 quadrants
    ✅ Processed: cate7-00014_jpg.rf.b6e7d5cf3469969cd7fdb18f45969c2f | Saved 4 quadrants
    ✅ Processed: cate8-00238_jpg.rf.788f2557da9def8a8e2ff843e4ce9e8e | Saved 4 quadrants
    ✅ Processed: cate7-00032_jpg.rf.7eb63fb11fac0b62f8af7efaa8194328 | Saved 4 quadrants
    ✅ Processed: cate4-00089_jpg.rf.6a2aeed3b1c4b9c2fcacff9124a98b63 | Saved 4 quadrants
    ✅ Processed: cate2-00033_jpg.rf.a17266f91a41813a3f83cf611dae1847 | Saved 4 quadrants
    ✅ Processed: cate6-00153_jpg.rf.f94a715c895a7cf7a599fd2e89debf93 | Saved 4 quadrants
    ✅ Processed: cate7-00069_jpg.rf.a356ede0d031c0626edf3735fc00bb93 | Saved 4 quadrants
    ✅ Processed: cate7-00110_jpg.rf.65944f41039a863ee29ef9b87843463d | Saved 4 quadrants
    ✅ Processed: cate2-00130_jpg.rf.a04c1ae2b52610b1a24547b3c5224f99 | Saved 4 quadrants
    Skipping filename: 633
    ✅ Processed: cate7-00056_jpg.rf.04bf152367412f58dbcd612a4429bda4 | Saved 4 quadrants
    Skipping filename: 633
    ✅ Processed: cate5-00025_jpg.rf.bfed93c6718d8e4fa12fdf802d73928d | Saved 2 quadrants
    ✅ Processed: cate10-00041_jpg.rf.d0b28afbdf3cd79d61728e1e50ba997e | Saved 4 quadrants
    ✅ Processed: cate8-00051_jpg.rf.dc01f9a2d611518110f41bf9f67a7ea9 | Saved 4 quadrants
    ✅ Processed: cate8-00143_jpg.rf.e68261a740f1013f5fa143688f513f13 | Saved 4 quadrants
    Skipping filename: 633
    ✅ Processed: cate8-00022_jpg.rf.6208f4e5ec4a46deba50a85d7c78ed46 | Saved 4 quadrants
    ✅ Processed: cate7-00112_jpg.rf.1b66981cb850d62bcd202c52d6ac303e | Saved 4 quadrants
    ✅ Processed: cate8-00001_jpg.rf.9d4edda05a747daa5731fe2b6807d334 | Saved 4 quadrants
    ✅ Processed: cate2-00128_jpg.rf.f98190fd7b358e0f89efb7202a2c3caa | Saved 4 quadrants
    ✅ Processed: cate2-00037_jpg.rf.af90517abe70a6b86d62746ce9be6ce3 | Saved 4 quadrants
    ✅ Processed: cate8-00041_jpg.rf.3f776daf5ea80c80e434a403c7baf77c | Saved 4 quadrants
    ✅ Processed: cate2-00006_jpg.rf.d99b5adc593a61cc345549ce7f3ded7b | Saved 4 quadrants
    Skipping filename: 633
    ✅ Processed: cate10-00108_jpg.rf.bb679aa97b13f1601c1926e499e25189 | Saved 4 quadrants
    ✅ Processed: cate6-00086_jpg.rf.3a82a7e3f578e2b20def54b9cce22356 | Saved 4 quadrants
    ✅ Processed: cate1-00038_jpg.rf.3c63a693eae24c5155b6f1282928630d | Saved 4 quadrants
    ✅ Processed: cate8-00173_jpg.rf.7b0e3c8a22a80631d5991c9b1609b8ca | Saved 4 quadrants
    Skipping filename: 633
    ✅ Processed: cate8-00181_jpg.rf.d5a041a8fb6cd04009afe799f84b10c2 | Saved 4 quadrants
    ✅ Processed: cate8-00073_jpg.rf.227f997e934b2d7ba5a29101620e8a57 | Saved 4 quadrants
    ✅ Processed: cate4-00103_jpg.rf.94b7fefebda62b27d129be8738b98299 | Saved 4 quadrants
    ✅ Processed: cate2-00029_jpg.rf.18be40d73e21136c043b185eece71e07 | Saved 4 quadrants
    ✅ Processed: cate8-00365_jpg.rf.4ac74d1d16385878c326c4b7777389c2 | Saved 4 quadrants
    ✅ Processed: cate7-00033_jpg.rf.4f58d6ee48fbac3831e4befe6b0c6660 | Saved 4 quadrants
    ✅ Processed: cate7-00011_jpg.rf.c69e41a99068ff6c739d526685d3c151 | Saved 4 quadrants
    ✅ Processed: cate8-00329_jpg.rf.90b7103bfb85c5374052b7f7161adba8 | Saved 4 quadrants
    ✅ Processed: cate6-00101_jpg.rf.27ed4471932f755f9bb621416bced5f5 | Saved 4 quadrants
    ✅ Processed: cate8-00053_jpg.rf.e3f5dcbd97cdc584dc8cb7fa89e54f24 | Saved 4 quadrants
    ✅ Processed: cate10-00092_jpg.rf.07a3664696ead2eca398494a634c93cf | Saved 4 quadrants
    Skipping filename: 633
    ✅ Processed: cate4-00094_jpg.rf.faf3f62b9faa8eb81853fe4452ae3604 | Saved 4 quadrants
    ✅ Processed: cate4-00107_jpg.rf.d28348f3fd74ad59bd11cdab2fbab85b | Saved 4 quadrants
    ✅ Processed: cate2-00103_jpg.rf.ec685b214407d3293af1801239821564 | Saved 4 quadrants
    ✅ Processed: cate7-00047_jpg.rf.5dcdd3a5b534dcd46e9b82dbc98c1d47 | Saved 4 quadrants
    ✅ Processed: cate2-00119_jpg.rf.415ee55c8d2cce1bad090e88328db634 | Saved 4 quadrants
    ✅ Processed: cate8-00066_jpg.rf.37b7b6313b22e517866d74c3d9a906ab | Saved 4 quadrants
    ✅ Processed: cate5-00001_jpg.rf.e4bb66bbdf06cd9feaeb0bb0f3cdd9f9 | Saved 1 quadrants
    ✅ Processed: cate2-00116_jpg.rf.d08949dc3fd667f83674ae6735292f10 | Saved 4 quadrants
    ✅ Processed: cate4-00131_jpg.rf.dceada043e33cbad1bc1b57ed1243bfc | Saved 4 quadrants
    ✅ Processed: cate8-00296_jpg.rf.95e9e4fa74799d3cdc335c02e04d23bf | Saved 4 quadrants
    ✅ Processed: cate7-00086_jpg.rf.ab178a3f5b034bae32a55b769faf7168 | Saved 4 quadrants
    ✅ Processed: cate6-00159_jpg.rf.d670e83aa91a1a86fe0596aa604d6bf4 | Saved 4 quadrants
    ✅ Processed: cate2-00032_jpg.rf.08ee0db4735d7ccc05e91c92b8a104d4 | Saved 4 quadrants
    ✅ Processed: cate5-00090_jpg.rf.98954c2a3b800e26af435cbe34114fa8 | Saved 4 quadrants
    ✅ Processed: cate8-00035_jpg.rf.f27d79e4dbe4679b9551d4c53d286221 | Saved 4 quadrants
    ✅ Processed: cate3-00013_jpg.rf.792f8537dbcd3ee15ae1bff55fdee873 | Saved 4 quadrants
    ✅ Processed: cate3-00018_jpg.rf.51a9baa986aa7438d3195a6b6e40f78e | Saved 4 quadrants
    ✅ Processed: cate8-00080_jpg.rf.748e9f718da334ce7b7fd4c912227da4 | Saved 4 quadrants
    ✅ Processed: cate8-00437_jpg.rf.ad26c049794c36d231e96951bd0482c6 | Saved 4 quadrants
    ✅ Processed: cate6-00063_jpg.rf.1de234c2cdc9079dd4ea04770d464153 | Saved 4 quadrants
    ✅ Processed: cate4-00102_jpg.rf.3f2e28e5dde71ef1bd67ee6ac392bbcb | Saved 4 quadrants
    ✅ Processed: cate7-00050_jpg.rf.d4270cdb911fe6fa3ede8caef193e6b1 | Saved 4 quadrants
    ✅ Processed: cate8-00210_jpg.rf.976add877ebfcf75311ab074d9c88a52 | Saved 4 quadrants
    ✅ Processed: cate7-00020_jpg.rf.2531ebb50b4c5a73a4f454312be08822 | Saved 4 quadrants
    ✅ Processed: cate2-00114_jpg.rf.1187a09bcb3a05a2fe0cb942506602e3 | Saved 4 quadrants
    ✅ Processed: cate8-00317_jpg.rf.c1cb14a5a2affc177e329ee15edc977c | Saved 4 quadrants
    ✅ Processed: cate5-00048_jpg.rf.713045de3a42978a2ba4ece39654917e | Saved 4 quadrants
    ✅ Processed: cate6-00099_jpg.rf.66421231c2c5a935538c461b06d72087 | Saved 4 quadrants
    ✅ Processed: cate2-00031_jpg.rf.8072470d88ef5495882a7c4a73349b16 | Saved 4 quadrants
    ✅ Processed: cate6-00039_jpg.rf.94cba00684b88294b4926d2ead2f577e | Saved 4 quadrants
    ✅ Processed: cate1-00053_jpg.rf.b2b1daf8086bd0629a7d3b72b37ccaf6 | Saved 4 quadrants
    ✅ Processed: cate10-00006_jpg.rf.b74997f6dded56cc38eed5a381ef5047 | Saved 4 quadrants
    ✅ Processed: cate4-00113_jpg.rf.239bf24468fa8c46ccf721c6515d5b3a | Saved 4 quadrants
    ✅ Processed: cate2-00012_jpg.rf.0484e22a6f13c36d8fc5ae4bbbf568c6 | Saved 4 quadrants
    ✅ Processed: cate8-00253_jpg.rf.ea84a9f38e3dfb75aeda9c55b636661b | Saved 4 quadrants
    ✅ Processed: cate2-00027_jpg.rf.cf608bef7b2d8bb65a96dbc4d793ccd7 | Saved 4 quadrants
    Skipping filename: 633
    ✅ Processed: cate5-00108_jpg.rf.c225547cfa6751eaa2040eebd50c4295 | Saved 2 quadrants
    ✅ Processed: cate5-00049_jpg.rf.f7127b9dd1ceafba3de09469220011f9 | Saved 4 quadrants
    ✅ Processed: cate8-00071_jpg.rf.428172fb9a908c9ac241ffb58df3709a | Saved 4 quadrants
    ✅ Processed: cate8-00227_jpg.rf.223c6f3947f2a93775e124289bd76f21 | Saved 4 quadrants
    ✅ Processed: cate10-00113_jpg.rf.88e25cd5a826e102fec8f956ebc56c4e | Saved 2 quadrants
    ✅ Processed: cate8-00221_jpg.rf.9e31964ab6b159357e5e6c047dd4a38d | Saved 4 quadrants
    ✅ Processed: cate6-00144_jpg.rf.4ee6cf9847d3e2d6466425ba9dc9b860 | Saved 4 quadrants
    ✅ Processed: cate8-00293_jpg.rf.2ffa30334a6f21e3b8b37f949b769f7c | Saved 4 quadrants
    ✅ Processed: cate4-00069_jpg.rf.3c40af59574c0b84e63758676751923e | Saved 4 quadrants
    Skipping filename: 633
    ✅ Processed: cate10-00070_jpg.rf.16c0aa0f85b4226272b67ba8723cf5cf | Saved 4 quadrants
    ✅ Processed: cate8-00245_jpg.rf.9ac244e171656fdc1ce9ed529eff3e62 | Saved 4 quadrants
    ✅ Processed: cate2-00111_jpg.rf.9501b5f21a6741a69453c5660ce8353e | Saved 4 quadrants
    ✅ Processed: cate8-00251_jpg.rf.7bf674716e46049af88bbc6fd0177355 | Saved 4 quadrants
    ✅ Processed: cate5-00043_jpg.rf.ee042e0b1d6a8b3b369c9eb86b3889d9 | Saved 4 quadrants
    ✅ Processed: cate5-00030_jpg.rf.c5b81054ce093534f94022eb9b2a1545 | Saved 4 quadrants
    ✅ Processed: cate8-00418_jpg.rf.3c8ae33b104c9cf3a98c0340629b248d | Saved 4 quadrants
    ✅ Processed: cate8-00101_jpg.rf.15de1fa858753fa3fd206599bff901f8 | Saved 4 quadrants
    ✅ Processed: cate5-00006_jpg.rf.890bb4918809e942965bffc4454895af | Saved 2 quadrants
    ✅ Processed: cate6-00036_jpg.rf.3e8b730477e2ddea033e7f85464cd962 | Saved 4 quadrants
    ✅ Processed: cate1-00026_jpg.rf.365e2e2d1d708d69d19a27a09f0b05de | Saved 4 quadrants
    ✅ Processed: cate8-00361_jpg.rf.46e08ec0ba94e5fa72b83b9d34394d23 | Saved 4 quadrants
    ✅ Processed: cate2-00003_jpg.rf.97f51317ee5851f101839f10764c288a | Saved 4 quadrants
    Skipping filename: 633
    ✅ Processed: cate2-00008_jpg.rf.10358f5cbbca8147ea334370e1ee92a6 | Saved 4 quadrants
    ✅ Processed: cate8-00281_jpg.rf.d80e055a68c129dd4fce709deb754d2c | Saved 4 quadrants
    ✅ Processed: cate4-00108_jpg.rf.89923db5b141a97f1a0ee29f3c1b9c9b | Saved 4 quadrants
    ✅ Processed: cate8-00246_jpg.rf.435262d802cfd76c1260aafa151feed9 | Saved 4 quadrants
    ✅ Processed: cate7-00074_jpg.rf.12a01f3dc902c5e705afe61f89aa0f44 | Saved 4 quadrants
    ✅ Processed: cate2-00035_jpg.rf.903bb19bbb06a11739abeb245ad6d4c9 | Saved 4 quadrants
    ✅ Processed: cate4-00095_jpg.rf.da7ee68b3c7d4726f45ebb0f3a2ee4cf | Saved 4 quadrants
    ✅ Processed: cate7-00092_jpg.rf.55f0968c3264a468d04e836e58ffa384 | Saved 4 quadrants
    ✅ Processed: cate3-00016_jpg.rf.d93c71c586be9b9c8ffb0e596b8904d1 | Saved 4 quadrants
    ✅ Processed: cate4-00105_jpg.rf.31fe9209cd5160d7535877a3b8d56e01 | Saved 4 quadrants
    ✅ Processed: cate8-00425_jpg.rf.23bcf560118705861b96153f6be57f6d | Saved 4 quadrants
    ✅ Processed: cate8-00269_jpg.rf.52b23eb0ac69e7b6a44858b3ac7d5ec6 | Saved 4 quadrants
    ✅ Processed: cate8-00209_jpg.rf.a6065d2c02bea08bd06e3de1030a0536 | Saved 4 quadrants
    ✅ Processed: cate9-00022_jpg.rf.b8b63c2f44c568efca270710d25ff5d2 | Saved 4 quadrants
    ✅ Processed: cate8-00119_jpg.rf.da38166875d4321c3e24ce39701807c3 | Saved 4 quadrants
    ✅ Processed: cate8-00390_jpg.rf.6488a2560a89a2f4e0dc1b2755b6421f | Saved 4 quadrants
    ✅ Processed: cate8-00197_jpg.rf.0b162fc7cdfe542463d39d52b7c82202 | Saved 4 quadrants
    ✅ Processed: cate1-00059_jpg.rf.a9e11268c4bba51e9373b827dc1b9753 | Saved 4 quadrants
    ✅ Processed: cate2-00030_jpg.rf.fa2addd1d466f0f70e5486aacaafbead | Saved 4 quadrants
    ✅ Processed: cate10-00077_jpg.rf.ec56afc535c753c1dec0416c28550d02 | Saved 4 quadrants
    ✅ Processed: cate4-00070_jpg.rf.9f0a5a0a2310b7fbcd76c6870d316074 | Saved 4 quadrants
    ✅ Processed: cate1-00029_jpg.rf.97fe17dca11f631d02b8290db8b4c8de | Saved 4 quadrants
    ✅ Processed: cate6-00158_jpg.rf.b53fd92ea4446991b66da5a3283ee406 | Saved 4 quadrants
    ✅ Processed: cate2-00022_jpg.rf.01f388558413301107d4a9f1a7a0d410 | Saved 4 quadrants
    ✅ Processed: cate8-00017_jpg.rf.d33a36a6e5743f8b8d81b77ca7901dba | Saved 4 quadrants
    ✅ Processed: cate2-00102_jpg.rf.65c9c658282d3cde127278893e6dc9c9 | Saved 4 quadrants
    ✅ Processed: cate8-00411_jpg.rf.2e2ceeedbe132938a911fefecf6ac3ea | Saved 4 quadrants
    ✅ Processed: cate7-00038_jpg.rf.febaeccf29df150d7273eb01633e9dee | Saved 4 quadrants
    Skipping filename: 633
    ✅ Processed: cate8-00159_jpg.rf.3d075740d264d12a6f4d0e3f8cf29445 | Saved 4 quadrants
    ✅ Processed: cate2-00122_jpg.rf.6e5ae3adde62c5a8c40a9de486c83b94 | Saved 4 quadrants
    ✅ Processed: cate3-00006_jpg.rf.aae79168d88bf4e20f6bfc51f3dd0f29 | Saved 4 quadrants
    ✅ Processed: cate8-00023_jpg.rf.69362c6be68b1d4e0f724f8f8e398f86 | Saved 4 quadrants
    ✅ Processed: cate8-00224_jpg.rf.51c1bcfede524df339233c145001ceb9 | Saved 4 quadrants
    ✅ Processed: cate8-00275_jpg.rf.460ddccaddfb64b4b229e2cbc45b8d92 | Saved 4 quadrants
    ✅ Processed: cate5-00007_jpg.rf.875910a4429adb2f8a817ac772717e58 | Saved 2 quadrants
    ✅ Processed: cate8-00083_jpg.rf.0ca9557ef935b20d9cb8d817ee00a6fb | Saved 4 quadrants
    ✅ Processed: cate2-00109_jpg.rf.4f7ba9a179030058bf19f2f8cb3958b1 | Saved 4 quadrants
    ✅ Processed: cate4-00082_jpg.rf.66780bdeb0ae89e078539db7e3e73524 | Saved 4 quadrants
    Skipping filename: 633
    ✅ Processed: cate1-00005_jpg.rf.59ad39ad4248e62f80b40fdc589e82df | Saved 4 quadrants
    ✅ Processed: cate2-00110_jpg.rf.819b67e36abf7ab9a1d7ad6f89b72767 | Saved 4 quadrants
    ✅ Processed: cate5-00031_jpg.rf.3e8b1c292e91d1bdf6752cb2ae16cf2d | Saved 4 quadrants
    ✅ Processed: cate2-00106_jpg.rf.c9b9875046320ef052a8807e49a365ca | Saved 4 quadrants
    ✅ Processed: cate2-00113_jpg.rf.13ff40ae6bbb7cddc3abb1bbda329c1e | Saved 4 quadrants
    ✅ Processed: cate4-00086_jpg.rf.5cf34b9934336f1da2d0cc111733f200 | Saved 4 quadrants
    ✅ Processed: cate5-00097_jpg.rf.a6d8f63884772b8fdb1668fdc3bdbcab | Saved 4 quadrants
    ✅ Processed: cate2-00007_jpg.rf.d7753ecf517d9a439fe7de643301610f | Saved 4 quadrants
    ✅ Processed: cate5-00024_jpg.rf.02951956ca210fbfaabbfb56e7e86a68 | Saved 4 quadrants
    ✅ Processed: cate4-00085_jpg.rf.a7d4df905289220b2fa91fe0b672cf19 | Saved 4 quadrants
    ✅ Processed: cate8-00440_jpg.rf.a6ee2ea22ff03515ba9f9a19203532bc | Saved 4 quadrants
    ✅ Processed: cate8-00107_jpg.rf.8c669c1f58479b7e33509c981a071843 | Saved 4 quadrants
    ✅ Processed: cate7-00018_jpg.rf.4ad7fa64ce5feba3d1b576d1ef00a112 | Saved 4 quadrants
    ✅ Processed: cate8-00375_jpg.rf.d85f5051e65719dc76fefba3d37b291e | Saved 4 quadrants
    ✅ Processed: cate5-00115_jpg.rf.481409bf4dfed95590e0ec97ff2f0dd6 | Saved 4 quadrants
    Skipping filename: 633
    ✅ Processed: cate2-00107_jpg.rf.38bbe97f6ee9a126926b3e50cfeff7f7 | Saved 4 quadrants
    ✅ Processed: cate3-00007_jpg.rf.b918c1031f894027acaa12ad1f8a8142 | Saved 4 quadrants
    ✅ Processed: cate8-00282_jpg.rf.696adf682e8a0e1dc89fdc079608b782 | Saved 4 quadrants
    ✅ Processed: cate6-00108_jpg.rf.e852ad137ec0ed547f51d8114d6a4844 | Saved 4 quadrants
    ✅ Processed: cate4-00112_jpg.rf.8888c76673fdb18c3510b7fe28dab826 | Saved 4 quadrants
    ✅ Processed: cate5-00036_jpg.rf.560839067a44932020f5a0d9134a9091 | Saved 4 quadrants
    ✅ Processed: cate8-00155_jpg.rf.e12bc1f48bd4fcc543b5f9c0124698ab | Saved 4 quadrants
    ✅ Processed: cate7-00105_jpg.rf.c78ec07e8fd0e62ebf9cdabca0ec4770 | Saved 4 quadrants
    ✅ Processed: cate2-00019_jpg.rf.bbf854cfb5bb343ba6faca4993eebca6 | Saved 4 quadrants
    ✅ Processed: cate2-00104_jpg.rf.eaa5f028720c00a5d75a34fdd945799a | Saved 4 quadrants
    ✅ Processed: cate10-00018_jpg.rf.e2fee68ab8c4d58743e6099bb6ccb597 | Saved 4 quadrants
    ✅ Processed: cate8-00116_jpg.rf.80c642117e3fa25349af9b929da9095a | Saved 2 quadrants
    ✅ Processed: cate8-00015_jpg.rf.ad9871585214d333e1ed323e40d13f2c | Saved 4 quadrants
    ✅ Processed: cate8-00044_jpg.rf.a69aebc8cbfdd1f94e0d317baea0dcc4 | Saved 4 quadrants
    ✅ Processed: cate5-00012_jpg.rf.fdbde98053c3d34af3ef17790786ff3b | Saved 4 quadrants
    ✅ Processed: cate2-00028_jpg.rf.e24bd2cce2923ea431883eec5de1da5f | Saved 4 quadrants
    ✅ Processed: cate4-00072_jpg.rf.0fbed3f154747c201c7c216b2634bd66 | Saved 4 quadrants
    ✅ Processed: cate8-00231_jpg.rf.2486fb8e48ad86d48572207f2b0d72f8 | Saved 4 quadrants
    ✅ Processed: cate10-00072_jpg.rf.103bc7aee3af6b953a340774e274ee2b | Saved 4 quadrants
    ✅ Processed: cate8-00152_jpg.rf.1554096b21f0bcd0fb09e83dd41602f1 | Saved 4 quadrants
    ✅ Processed: cate2-00123_jpg.rf.f28cf0e30b2c60d6826177f572800fd6 | Saved 4 quadrants
    ✅ Processed: cate10-00084_jpg.rf.28a9a7a29e2435a28672832b6f727d59 | Saved 4 quadrants
    ✅ Processed: cate2-00038_jpg.rf.d2c9b3799254cc04594f41691f96629a | Saved 4 quadrants
    ✅ Processed: cate1-00014_jpg.rf.ec1e433bb6e41264fa4819877fe7486a | Saved 4 quadrants
    ✅ Processed: cate8-00371_jpg.rf.30602f943e8a893fe43b7c5e1f98ea1a | Saved 4 quadrants
    ✅ Processed: cate8-00323_jpg.rf.8f0f641400afdb2aaaeb3426b1da0807 | Saved 4 quadrants
    ✅ Processed: cate5-00084_jpg.rf.bf231961b5cf86ba29e10c1a195514b6 | Saved 4 quadrants
    ✅ Processed: cate5-00042_jpg.rf.186d28f60fa957912fab787b0974cf5b | Saved 4 quadrants
    ✅ Processed: cate3-00019_jpg.rf.99434bff32d2808f7633e6b301ab854f | Saved 4 quadrants
    ✅ Processed: cate2-00136_jpg.rf.5a4262b479d47c5fb50d58531ed744d2 | Saved 4 quadrants
    ✅ Processed: cate5-00079_jpg.rf.fd794c1ff311638c1bf9d35661acd1d2 | Saved 4 quadrants
    ✅ Processed: cate2-00005_jpg.rf.0c97882aaa66d6f3a45ea997755cde65 | Saved 4 quadrants
    ✅ Processed: cate1-00041_jpg.rf.08c050565c2241e65e4ca3ce7f714368 | Saved 4 quadrants
    ✅ Processed: cate8-00354_jpg.rf.c24ef893d230f9667bac4236b11f2ee6 | Saved 4 quadrants
    ✅ Processed: cate8-00174_jpg.rf.08ecaca067424326827c55f7e4465e02 | Saved 4 quadrants
    ✅ Processed: cate8-00332_jpg.rf.7287c0bf54df93c2458263fe8b902a29 | Saved 4 quadrants
    ✅ Processed: cate5-00096_jpg.rf.442eb55bbdc18b6aff7f7306862cf2fe | Saved 4 quadrants
    ✅ Processed: cate10-00042_jpg.rf.ac4c7d9a140004c1478deba941a7d750 | Saved 4 quadrants
    ✅ Processed: cate2-00014_jpg.rf.0091b35af2de3ee57c6a3c68477dff0c | Saved 4 quadrants
    Skipping filename: 633
    ✅ Processed: cate6-00009_jpg.rf.cb2f9d02d718a9fbe6515a74e7a0728d | Saved 4 quadrants
    ✅ Processed: cate1-00047_jpg.rf.bcf913fc3b431f25462c25c32c8435d7 | Saved 4 quadrants
    ✅ Processed: cate8-00260_jpg.rf.d03427019b70094be862d65d92451c32 | Saved 4 quadrants
    ✅ Processed: cate8-00454_jpg.rf.12dc67e8e745ef1184b3738d438a3c8a | Saved 4 quadrants
    ✅ Processed: cate8-00185_jpg.rf.956835f4c90d8fe1295d6fdcdf9e7619 | Saved 4 quadrants
    ✅ Processed: cate1-00068_jpg.rf.f9e5e1d821dd0c82fe3224544a745710 | Saved 4 quadrants
    ✅ Processed: cate8-00095_jpg.rf.a2ef696b9a710dfc8e1ddadc6e124347 | Saved 4 quadrants
    ✅ Processed: cate7-00008_jpg.rf.73d8483560b93e61974c5bd0cb8e15b0 | Saved 4 quadrants
    Skipping filename: 633
    ✅ Processed: cate7-00026_jpg.rf.6bb14c357a56aae1898617ff4d88e00e | Saved 4 quadrants
    ✅ Processed: cate8-00353_jpg.rf.97c9376cc036a3b7a291028278e4d9ea | Saved 4 quadrants
    ✅ Processed: cate6-00075_jpg.rf.04c5dcc3e38725ebba32acd18aa69be8 | Saved 4 quadrants
    ✅ Processed: cate8-00335_jpg.rf.cde34b5580804d14e0d0a53c2774526c | Saved 4 quadrants
    ✅ Processed: cate8-00109_jpg.rf.e44bfe4207af86e7d7ce5ea2bd2db298 | Saved 4 quadrants
    ✅ Processed: cate2-00016_jpg.rf.f946abb272ede06f67d1102c72c9cc8a | Saved 4 quadrants
    ✅ Processed: cate4-00104_jpg.rf.940afd43ec0f7710998e2aa2afd6ad39 | Saved 4 quadrants
    ✅ Processed: cate7-00062_jpg.rf.d09401206a92456ef1e617de6991a3e4 | Saved 4 quadrants
    ✅ Processed: cate7-00076_jpg.rf.cd7d55b244aef8143b825ddf1669b2be | Saved 4 quadrants
    Skipping filename: 633
    ✅ Processed: cate10-00096_jpg.rf.847ccd33e35016442a58b3832dfcc488 | Saved 4 quadrants
    Skipping filename: 633
    ✅ Processed: cate7-00104_jpg.rf.f776b4064893c3c43d0af221c79478cf | Saved 4 quadrants
    ✅ Processed: cate8-00382_jpg.rf.b15b5fc2ba64ab191e962dde3fef7395 | Saved 4 quadrants
    Skipping filename: 633
    ✅ Processed: cate10-00063_jpg.rf.94670f63657ade16a19bf7d454ecea01 | Saved 4 quadrants
    ✅ Processed: cate2-00025_jpg.rf.dc865af8fb9f78a9586c217e61b0c1ac | Saved 4 quadrants
    ✅ Processed: cate5-00072_jpg.rf.c231300ac6c61afdb3b2bbb82736baa3 | Saved 4 quadrants
    ✅ Processed: cate2-00015_jpg.rf.4f900758238cf313599b7ed8600214c8 | Saved 4 quadrants
    ✅ Processed: cate5-00102_jpg.rf.bacc49b0f354808df1e427b4e692ad88 | Saved 2 quadrants
    Skipping filename: 633
    Skipping filename: 633
    ✅ Processed: cate5-00067_jpg.rf.eb8ae2c0628eb1a3e6592c2b9b034fd8 | Saved 4 quadrants
    ✅ Processed: cate3-00023_jpg.rf.c0a91bde7954b7a7bb9794469862378a | Saved 4 quadrants
    ✅ Processed: cate2-00117_jpg.rf.6e1ff77390e8fd69fe7e24b569c486ed | Saved 4 quadrants
    Skipping filename: 633
    ✅ Processed: cate9-00016_jpg.rf.71e7d819e23adb49f9a81da637c1a021 | Saved 4 quadrants
    ✅ Processed: cate2-00101_jpg.rf.9f60235642c365cc56434856eef4ee09 | Saved 4 quadrants
    ✅ Processed: cate10-00005_jpg.rf.c49013bcaa37725c1bc8b2380f7fe4e3 | Saved 4 quadrants
    ✅ Processed: cate10-00024_jpg.rf.e7c951f564a56d45f88c13401e4ccfdb | Saved 4 quadrants
    ✅ Processed: cate8-00433_jpg.rf.d2de4b621c8e1324dfa0f74674070cc3 | Saved 4 quadrants
    ✅ Processed: cate8-00413_jpg.rf.213add826f8153ddae13737dc8913817 | Saved 4 quadrants
    ✅ Processed: cate2-00026_jpg.rf.52f53f396eede9b33d3f2d0c4c26d419 | Saved 4 quadrants
    ✅ Processed: cate4-00091_jpg.rf.a305343401dd3ee09d9cb4f5395fa184 | Saved 4 quadrants
    ✅ Processed: cate9-00028_jpg.rf.efc51f48960760aff64535936dd7669c | Saved 4 quadrants
    ✅ Processed: cate8-00377_jpg.rf.102c0ff39ae9e055d8bb4f76dffaa85e | Saved 4 quadrants
    ✅ Processed: cate8-00339_jpg.rf.488d6190a2e29e6b42460853a638983c | Saved 4 quadrants
    ✅ Processed: cate5-00103_jpg.rf.bd1da1ffacb4fb36a3f4e644ab117b11 | Saved 4 quadrants
    ✅ Processed: cate1-00062_jpg.rf.35e5b385c0d5970753e864741be4bf1f | Saved 4 quadrants
    ✅ Processed: cate8-00011_jpg.rf.c1505231652c384d6c4796287ade1cfb | Saved 4 quadrants
    ✅ Processed: cate10-00027_jpg.rf.dcef59f7280d31971dfa021de87c46d6 | Saved 4 quadrants
    ✅ Processed: cate9-00004_jpg.rf.89c18db03607eaf502cd61c92877e251 | Saved 4 quadrants
    ✅ Processed: cate7-00098_jpg.rf.2bf50d90633f155086b34495c9d6f562 | Saved 4 quadrants
    Skipping filename: 633
    ✅ Processed: cate8-00443_jpg.rf.35063d3dec7ab93e4e2b9361552342b6 | Saved 4 quadrants
    ✅ Processed: cate7-00097_jpg.rf.95e4d33330de7632ce7d1409c3240706 | Saved 4 quadrants
    ✅ Processed: cate8-00267_jpg.rf.448731e1dbac9e27b7874299936762dc | Saved 4 quadrants
    ✅ Processed: cate4-00075_jpg.rf.c1a9139dafc147b05de83f14df06b083 | Saved 4 quadrants
    ✅ Processed: cate10-00078_jpg.rf.41e9196b8b1d01a990eaf842bb5c273e | Saved 1 quadrants
    ✅ Processed: cate8-00447_jpg.rf.f7bb88cbe24ed7b983a5e3713165d75c | Saved 4 quadrants
    ✅ Processed: cate8-00404_jpg.rf.402e14292cdaf9dd1b4e68ca94baaae4 | Saved 4 quadrants
    ✅ Processed: cate8-00215_jpg.rf.82010df7c6366f93700f04dd4e7bbdc6 | Saved 4 quadrants
    ✅ Processed: cate7-00044_jpg.rf.ae1cd65e6b07ed1d1aa580a7c3b01920 | Saved 4 quadrants
    ✅ Processed: cate6-00069_jpg.rf.1d2b55062ed75dda29be8f13083f59a0 | Saved 4 quadrants
    Skipping filename: 633
    ✅ Processed: cate2-00105_jpg.rf.45cac4248100d4856cc4a83d20298685 | Saved 4 quadrants
    ✅ Processed: cate1-00044_jpg.rf.b36e4113555ae485a444f914baf9ec12 | Saved 4 quadrants
    ✅ Processed: cate4-00079_jpg.rf.d9df3614b2f590b9e5510ac539049249 | Saved 4 quadrants
    ✅ Processed: cate2-00009_jpg.rf.04cf3ada666bdd2e73ca64960cd6baaa | Saved 4 quadrants
    ✅ Processed: cate6-00022_jpg.rf.174b97655544845cc0a0f9c83df52d7a | Saved 4 quadrants
    ✅ Processed: cate2-00135_jpg.rf.e3988f97c01947120c42d2477f751e97 | Saved 4 quadrants
    ✅ Processed: cate8-00123_jpg.rf.28b8f4b0e00a7ff7084ef2052da86c17 | Saved 4 quadrants
    ✅ Processed: cate7-00068_jpg.rf.89180b06ee83a983a161563b1fc465f2 | Saved 4 quadrants
    ✅ Processed: cate5-00078_jpg.rf.1782318284b60f49c02ed07f453566bd | Saved 2 quadrants
    ✅ Processed: cate4-00106_jpg.rf.0b94a0c5f3ce997322d764ff6f7eca16 | Saved 4 quadrants
    ✅ Processed: cate5-00037_jpg.rf.9fec1ae9aa6fa2275c42f92252750104 | Saved 4 quadrants
    ✅ Processed: cate8-00047_jpg.rf.921d5a76e559c3eaad17172814d9d8cf | Saved 4 quadrants
    ✅ Processed: cate2-00133_jpg.rf.80fe64612c7cf667a459c51e95936dc8 | Saved 4 quadrants
    ✅ Processed: cate1-00035_jpg.rf.e82433024ad10ea2a258614c2a8d3c5f | Saved 4 quadrants
    ✅ Processed: cate8-00389_jpg.rf.fca91494857d0715664a085b74607ad1 | Saved 4 quadrants
    ✅ Processed: cate2-00126_jpg.rf.ce79bbef8c026e2d73fcc5d177e34588 | Saved 4 quadrants
    ✅ Processed: cate1-00008_jpg.rf.10436fa8d35c0764df5402f3a6f093dc | Saved 4 quadrants
    ✅ Processed: cate6-00087_jpg.rf.e235d20610b41e54729c8535ce4fab72 | Saved 4 quadrants
    ✅ Processed: cate10-00085_jpg.rf.109e6668ebf8011512eff39c1e5a775a | Saved 2 quadrants
    ✅ Processed: cate4-00100_jpg.rf.0b9e55f417166a10342b47142e3928b0 | Saved 4 quadrants
    ✅ Processed: cate10-00090_jpg.rf.56cc8f337841b2f5d5c6b59eaff05e3b | Saved 4 quadrants
    ✅ Processed: cate2-00034_jpg.rf.35a0cbab89b1a5230413673468ade569 | Saved 4 quadrants
    ✅ Processed: cate3-00020_jpg.rf.10cddb9724f110337a80a95e65753161 | Saved 4 quadrants
    ✅ Processed: cate2-00124_jpg.rf.b94378b347005894adbaef813fe8dc76 | Saved 4 quadrants
    ✅ Processed: cate6-00105_jpg.rf.8bfca665aed9ad8bd5871d71fc09e5f5 | Saved 4 quadrants
    ✅ Processed: cate1-00056_jpg.rf.e32ad1ffb82e718b99dd09c52782890d | Saved 4 quadrants
    ✅ Processed: cate5-00018_jpg.rf.2df8e1e8e2d18d481102fe3db411dcaa | Saved 4 quadrants
    ✅ Processed: cate9-00040_jpg.rf.ac665e4a5574a7653c1d0bdcf0c028b4 | Saved 4 quadrants
    ✅ Processed: cate8-00037_jpg.rf.ecc698db6897d3b0c50ea1e901b6df2d | Saved 4 quadrants
    ✅ Processed: cate2-00002_jpg.rf.88df00cce5a1d60ebf0e6b7fd48b8e71 | Saved 4 quadrants
    ✅ Processed: cate8-00310_jpg.rf.9ca875396f270366be21b2d4f70a1253 | Saved 4 quadrants
    ✅ Processed: cate8-00094_jpg.rf.c7c830a886d59fb196ffed83200754d7 | Saved 4 quadrants
    ✅ Processed: cate2-00120_jpg.rf.41925bd41fb5fa6be214f35064de5dbc | Saved 4 quadrants
    ✅ Processed: cate3-00014_jpg.rf.8042132daeee070cd7bf4fb49c14f85a | Saved 4 quadrants
    ✅ Processed: cate2-00024_jpg.rf.55ee5e910d8c2bfb220cd2f31f5738c8 | Saved 4 quadrants
    ✅ Processed: cate8-00401_jpg.rf.2a8f186411ae92fb54571575a8b88368 | Saved 4 quadrants
    ✅ Processed: cate8-00059_jpg.rf.420768f9135f713d1965159d0ca6c3e5 | Saved 4 quadrants
    ✅ Processed: cate4-00099_jpg.rf.710e9edac74fdf77a1a2feb13d809232 | Saved 4 quadrants
    ✅ Processed: cate8-00347_jpg.rf.b91e56f649425c08edd69986ca9c971e | Saved 4 quadrants
    ✅ Processed: cate4-00088_jpg.rf.d5f9b0e2f64eb033dcf8f86c5f0374a5 | Saved 4 quadrants
    ✅ Processed: cate1-00065_jpg.rf.7a710e6eddcb3ced08814a62a93143a6 | Saved 4 quadrants
    ✅ Processed: cate8-00397_jpg.rf.8ccc23412af94ef90976e09ce1f4982e | Saved 4 quadrants
    ✅ Processed: cate10-00036_jpg.rf.ea49679c987ad05267c5c26db898aaa5 | Saved 4 quadrants
    ✅ Processed: cate8-00311_jpg.rf.01c8c4b6e41f02c88a473ea853f07c50 | Saved 4 quadrants
    ✅ Processed: cate8-00359_jpg.rf.9dbe2c04d7e56a9ead4e28c58268c748 | Saved 4 quadrants
    ✅ Processed: cate5-00066_jpg.rf.5d26331c0b7cadb2d96e382b7e7332e9 | Saved 4 quadrants
    ✅ Processed: cate6-00122_jpg.rf.83a34eeb48a30429e949b04145c8b4aa | Saved 4 quadrants
    ✅ Processed: cate1-00002_jpg.rf.e0c956c6468ef96b53c862916e6fb6e8 | Saved 4 quadrants
    ✅ Processed: cate8-00065_jpg.rf.1950a5f867fd319f4085138a13bf80cb | Saved 4 quadrants
    ✅ Processed: cate8-00166_jpg.rf.76201a51aaebc0e1813c8d203a2deafa | Saved 4 quadrants
    ✅ Processed: cate8-00287_jpg.rf.d3c67f5ba97dc9f85f69c6f6f9c5a8c7 | Saved 4 quadrants
    ✅ Processed: cate3-00021_jpg.rf.aeacd9839f0cba986b922228ef2183bc | Saved 4 quadrants
    ✅ Processed: cate8-00149_jpg.rf.74b75219fd6c9070f717103be835a675 | Saved 4 quadrants
    ✅ Processed: cate8-00191_jpg.rf.856a6afd259fc24434201fdf1c9b37a0 | Saved 2 quadrants
    ✅ Processed: cate2-00129_jpg.rf.f341f1cc565a40e9be61f8ba8cf2d12c | Saved 4 quadrants
    Skipping filename: 633
    ✅ Processed: cate1-00071_jpg.rf.de9b6e67d43a02106a0cb46bc49fb0ad | Saved 4 quadrants
    ✅ Processed: cate8-00008_jpg.rf.2cc888fee5099703c3037e129ea399ac | Saved 4 quadrants
    ✅ Processed: cate10-00114_jpg.rf.0b4b5fb255f359f67147cbba792fa32b | Saved 4 quadrants
    ✅ Processed: cate8-00138_jpg.rf.812b92a6c7752d24cef67b74d8a01045 | Saved 4 quadrants
    ✅ Processed: cate8-00239_jpg.rf.3640ab2eb42bcf1f2960990159b79eaf | Saved 4 quadrants
    ✅ Processed: cate8-00188_jpg.rf.3d20e8e90fa37e6d2761841f47fa16b2 | Saved 4 quadrants
    ✅ Processed: cate2-00134_jpg.rf.176f9f3dc7fcfcc35fb2abbdd10065e3 | Saved 4 quadrants
    ✅ Processed: cate6-00115_jpg.rf.4c525a1e07293b84ad8ee02893dccd9d | Saved 4 quadrants
    ✅ Processed: cate8-00257_jpg.rf.a338f126164098cc47f0480280f95dd2 | Saved 4 quadrants
    ✅ Processed: cate8-00167_jpg.rf.2212a8c10ac470bbf18d9be06e6fb552 | Saved 4 quadrants
    ✅ Processed: cate10-00030_jpg.rf.d3e923451688ebd1865ebd7575569547 | Saved 4 quadrants
    ✅ Processed: cate8-00289_jpg.rf.26350e9552ec3bbc0ca382d617b7c1c6 | Saved 4 quadrants
    ✅ Processed: cate2-00004_jpg.rf.1fb15fe8ac6efd72e12e1eb323fe15c2 | Saved 4 quadrants
    ✅ Processed: cate8-00145_jpg.rf.8422a0e4de12ea2ba9da6632e1aa97ea | Saved 4 quadrants
    ✅ Processed: cate8-00077_jpg.rf.61b2c3b135dcdc56d184a1dac5d2f977 | Saved 4 quadrants
    ✅ Processed: cate10-00012_jpg.rf.cd660ac6ee607fc1e309c9ecf1c4884d | Saved 4 quadrants
    ✅ Processed: cate7-00080_jpg.rf.a2d872feea4a01def68aee2026aef100 | Saved 4 quadrants
    ✅ Processed: cate2-00118_jpg.rf.542d865e75a076fce0a3b52e95ff0b3e | Saved 4 quadrants
    ✅ Processed: cate2-00017_jpg.rf.5e6addfda949536cf79186d34733685a | Saved 4 quadrants
    ✅ Processed: cate8-00407_jpg.rf.d4d8a2efa2ee0879c0287c4ef370e123 | Saved 4 quadrants
    ✅ Processed: cate7-00083_jpg.rf.5039c430c9724ce99702fdf0815dcd6f | Saved 4 quadrants
    ✅ Processed: cate2-00108_jpg.rf.e840f24b9375263c05d361a616ec8022 | Saved 4 quadrants
    ✅ Processed: cate8-00274_jpg.rf.a1faa79540a89f57833fd41a51a48405 | Saved 4 quadrants
    ✅ Processed: cate3-00026_jpg.rf.e4c4b547dc19149926e48cbe99a5d8cd | Saved 4 quadrants
    ✅ Processed: cate8-00426_jpg.rf.2e1c303bd8ce9272d8d0493ec35033c3 | Saved 4 quadrants
    ✅ Processed: cate10-00034_jpg.rf.8d39c2630e2949a8c1e1e2d78a8101a7 | Saved 4 quadrants
    ✅ Processed: cate8-00233_jpg.rf.fccdf75c2c0e5b817e86a7be0d18ecad | Saved 4 quadrants
    ✅ Processed: cate4-00114_jpg.rf.ca966fe9e68f24c9582d86c1e0317e6a | Saved 4 quadrants
    ✅ Processed: cate8-00341_jpg.rf.64724001e9633ae9ed9dae0026b7d7e1 | Saved 4 quadrants
    ✅ Processed: cate2-00013_jpg.rf.8182c09690596e2b2d7873ff1811866f | Saved 4 quadrants
    ✅ Processed: cate8-00449_jpg.rf.e3d7c655fd2a01ec12b788edb9fd079f | Saved 2 quadrants
    ✅ Processed: cate8-00005_jpg.rf.6cf69d25c438988ebddc53c9c676bd31 | Saved 4 quadrants
    ✅ Processed: cate10-00102_jpg.rf.6b830e6892bd86f1df42b2289a80275f | Saved 4 quadrants
    ✅ Processed: cate10-00106_jpg.rf.87a979c10735a55fc4d04e3b2decac07 | Saved 4 quadrants
    ✅ Processed: cate8-00419_jpg.rf.9b8063931a5f0eca0de6c74f8f920049 | Saved 4 quadrants
    ✅ Processed: cate1-00050_jpg.rf.b3601c7dfe121061a892cd6f16b8b27a | Saved 4 quadrants
    ✅ Processed: cate6-00123_jpg.rf.c8e735c051fed483e84298a9cb8f17b3 | Saved 4 quadrants
    ✅ Processed: cate8-00263_jpg.rf.e2948b82ed7a240f4b20fbfe79864440 | Saved 4 quadrants
    ✅ Processed: cate2-00039_jpg.rf.6e30ed8c13e37e37aee804a111dea8e0 | Saved 4 quadrants
    ✅ Processed: cate9-00010_jpg.rf.0fe694399f8c36de4f138b2a9a62f736 | Saved 4 quadrants
    ✅ Processed: cate8-00161_jpg.rf.518b888d0c5c33ae085dc93d711bf435 | Saved 4 quadrants
    ✅ Processed: cate5-00055_jpg.rf.f5ab103c5396aeb1587b0fd641ffb289 | Saved 1 quadrants
    ✅ Processed: cate8-00346_jpg.rf.11b061e808a839a3a030b80256b840f0 | Saved 4 quadrants
    ✅ Processed: cate8-00130_jpg.rf.9078c91ea833c034264c01bc600f65c2 | Saved 4 quadrants
    ✅ Processed: cate8-00455_jpg.rf.a7c58894213e16c969726e244c5b10f5 | Saved 4 quadrants
    ✅ Processed: cate8-00368_jpg.rf.8ab3f1ae318692e580edbe5dbb527b5c | Saved 4 quadrants
    ✅ Processed: cate8-00087_jpg.rf.e632d3538a2e68e1a69f3efc72273473 | Saved 4 quadrants
    ✅ Processed: cate8-00058_jpg.rf.d31b88e25962a40a2ef50b160c2bbdbe | Saved 4 quadrants
    ✅ Processed: cate8-00383_jpg.rf.ae2c13dba01cfae2c7928315b0c128a4 | Saved 4 quadrants
    ✅ Processed: cate10-00049_jpg.rf.19ce72ce93d19194b5979f4f55d9a916 | Saved 4 quadrants
    ✅ Processed: cate5-00073_jpg.rf.35ef24b02f2a62a2073e949f9ddf047f | Saved 4 quadrants
    ✅ Processed: cate2-00011_jpg.rf.ad2efe7bc151b5cfcf887b6b63e8de0b | Saved 4 quadrants
    ✅ Processed: cate5-00085_jpg.rf.ae53af885087d3e0b424f12c4f2c1357 | Saved 4 quadrants
    ✅ Processed: cate1-00020_jpg.rf.39ef8bdd86011ce21e3b0e9b6eaa82b6 | Saved 4 quadrants
    ✅ Processed: cate8-00125_jpg.rf.fea9db534d4f5069646acaa76d650c8d | Saved 4 quadrants
    ✅ Processed: cate2-00040_jpg.rf.611a724ab77f4f0b58a3872c672bc109 | Saved 4 quadrants
    ✅ Processed: cate7-00025_jpg.rf.00d895357624d877d0ad3b23e1dcc34d | Saved 4 quadrants
    ✅ Processed: cate8-00203_jpg.rf.a19549725499919b0628681946ce1d80 | Saved 4 quadrants
    ✅ Processed: cate8-00217_jpg.rf.4239ab7eb56769ea00ffe002e21f10b8 | Saved 4 quadrants
    ✅ Processed: cate10-00013_jpg.rf.6611586688f26e8e71280fb4c94ae494 | Saved 4 quadrants
    ✅ Processed: cate5-00061_jpg.rf.06b2bd4f9d9083fbab3b38698c67dba4 | Saved 4 quadrants
    ✅ Processed: cate6-00027_jpg.rf.f527451eb9f5af83370ee7c4211d5b00 | Saved 4 quadrants
    ✅ Processed: cate8-00305_jpg.rf.8c8278b3305348748e46af2f33e2e538 | Saved 4 quadrants
    ✅ Processed: cate4-00073_jpg.rf.42e2dae6771a43bf524a930eb73e6ebb | Saved 4 quadrants
    ✅ Processed: cate2-00112_jpg.rf.2acee4ae1d94cb4342a95211a91f401b | Saved 4 quadrants
    ✅ Processed: cate8-00137_jpg.rf.34823f8760ec5c1519d7ca8cdc4c4de2 | Saved 4 quadrants
    ✅ Processed: cate2-00115_jpg.rf.feb85e64865d69bce1e36077deaeab61 | Saved 4 quadrants
    ✅ Processed: cate2-00036_jpg.rf.5246ad8e1dc36f55114045c59978b45b | Saved 4 quadrants
    ✅ Processed: cate3-00027_jpg.rf.9a432908669dc9b82a86ff232a55ff45 | Saved 4 quadrants
    ✅ Processed: cate8-00102_jpg.rf.cee7888c447da5759b13399d44a0637c | Saved 4 quadrants
    ✅ Processed: cate8-00089_jpg.rf.5851642e5be77ef0f87b5df7a27a6e75 | Saved 4 quadrants
    ✅ Processed: cate7-00061_jpg.rf.e2d12b0de88f3f76ead612a948131a10 | Saved 4 quadrants
    Skipping filename: 633
    ✅ Processed: cate2-00021_jpg.rf.a7766bb31902e0bc793a6a34a9c29284 | Saved 4 quadrants
    ✅ Processed: cate1-00011_jpg.rf.8a5912281957cd055b390274289bde8e | Saved 4 quadrants
    ✅ Processed: cate8-00303_jpg.rf.a3e701bff3c669333acfc92852546815 | Saved 4 quadrants
    ✅ Processed: cate1-00017_jpg.rf.64ff6967f5d736e8a146556885ff4494 | Saved 4 quadrants
    ✅ Processed: cate10-00020_jpg.rf.854908ccabca768285c56c1fe9a7981a | Saved 4 quadrants
    ✅ Processed: cate2-00018_jpg.rf.5727e4a622c59be60ccffa85fa092acc | Saved 4 quadrants
    ✅ Processed: cate3-00003_jpg.rf.242fe63ad7de085f4641dfa555cefc93 | Saved 4 quadrants
    ✅ Processed: cate8-00318_jpg.rf.9c38532c41038b46ed0c9cbc143e63d5 | Saved 4 quadrants
    ✅ Processed: cate3-00008_jpg.rf.b75dae6c62ee59c4cc299ad5c7c458b7 | Saved 4 quadrants
    ✅ Processed: cate1-00023_jpg.rf.a14af26ab3a09c81f16c707f6f0daa53 | Saved 4 quadrants
    ✅ Processed: cate8-00325_jpg.rf.8634aff82b7f16dfc422b11471dda8c2 | Saved 4 quadrants
    ✅ Processed: cate2-00020_jpg.rf.1c2a690fd0d482c589834bd4a609d86c | Saved 4 quadrants
    Skipping filename: 633
    ✅ Processed: cate4-00116_jpg.rf.99afef2d5c49533ec4a4912063dee9cf | Saved 4 quadrants
    ✅ Processed: cate4-00077_jpg.rf.26f7c103bb785f607d093996c2377a22 | Saved 4 quadrants



```python
for label, image in zip(all_labels, all_images):
    if image in not_include_idx:
        plot_full_diagnostic(image, label, padding=80)
```


    
![png](notebooks/04_1_teeth_detection_dataset_files/notebooks/04_1_teeth_detection_dataset_29_0.png)
    



    
![png](notebooks/04_1_teeth_detection_dataset_files/notebooks/04_1_teeth_detection_dataset_29_1.png)
    



    
![png](notebooks/04_1_teeth_detection_dataset_files/notebooks/04_1_teeth_detection_dataset_29_2.png)
    



    
![png](notebooks/04_1_teeth_detection_dataset_files/notebooks/04_1_teeth_detection_dataset_29_3.png)
    



    
![png](notebooks/04_1_teeth_detection_dataset_files/notebooks/04_1_teeth_detection_dataset_29_4.png)
    



    
![png](notebooks/04_1_teeth_detection_dataset_files/notebooks/04_1_teeth_detection_dataset_29_5.png)
    



    
![png](notebooks/04_1_teeth_detection_dataset_files/notebooks/04_1_teeth_detection_dataset_29_6.png)
    



    
![png](notebooks/04_1_teeth_detection_dataset_files/notebooks/04_1_teeth_detection_dataset_29_7.png)
    



    
![png](notebooks/04_1_teeth_detection_dataset_files/notebooks/04_1_teeth_detection_dataset_29_8.png)
    



    
![png](notebooks/04_1_teeth_detection_dataset_files/notebooks/04_1_teeth_detection_dataset_29_9.png)
    



    
![png](notebooks/04_1_teeth_detection_dataset_files/notebooks/04_1_teeth_detection_dataset_29_10.png)
    



    
![png](notebooks/04_1_teeth_detection_dataset_files/notebooks/04_1_teeth_detection_dataset_29_11.png)
    



    
![png](notebooks/04_1_teeth_detection_dataset_files/notebooks/04_1_teeth_detection_dataset_29_12.png)
    



    
![png](notebooks/04_1_teeth_detection_dataset_files/notebooks/04_1_teeth_detection_dataset_29_13.png)
    



    
![png](notebooks/04_1_teeth_detection_dataset_files/notebooks/04_1_teeth_detection_dataset_29_14.png)
    



    
![png](notebooks/04_1_teeth_detection_dataset_files/notebooks/04_1_teeth_detection_dataset_29_15.png)
    



    
![png](notebooks/04_1_teeth_detection_dataset_files/notebooks/04_1_teeth_detection_dataset_29_16.png)
    



    
![png](notebooks/04_1_teeth_detection_dataset_files/notebooks/04_1_teeth_detection_dataset_29_17.png)
    



    
![png](notebooks/04_1_teeth_detection_dataset_files/notebooks/04_1_teeth_detection_dataset_29_18.png)
    



    
![png](notebooks/04_1_teeth_detection_dataset_files/notebooks/04_1_teeth_detection_dataset_29_19.png)
    



    
![png](notebooks/04_1_teeth_detection_dataset_files/notebooks/04_1_teeth_detection_dataset_29_20.png)
    



    
![png](notebooks/04_1_teeth_detection_dataset_files/notebooks/04_1_teeth_detection_dataset_29_21.png)
    



    
![png](notebooks/04_1_teeth_detection_dataset_files/notebooks/04_1_teeth_detection_dataset_29_22.png)
    



```python
for label_dict, image_path in zip(all_labels, all_images):
    if image_path in not_include_idx:
        prepare_stage2_data(image_path, label_dict)
```

    ✅ Processed: cate10-00060_jpg.rf.49a69a671bd365a7282c5a449ede5306 | Saved 4 quadrants
    ✅ Processed: cate6-00147_jpg.rf.a1ad961c2145387bf9233353abd43f72 | Saved 4 quadrants
    ✅ Processed: cate6-00065_jpg.rf.2c543260f3d2116dc86c6bac2da6b296 | Saved 4 quadrants
    ✅ Processed: cate6-00014_jpg.rf.7f16a15b6dfcdc9c6faaaba7a56a8f13 | Saved 4 quadrants
    ✅ Processed: cate2-00023_jpg.rf.6a182c232906217c5f14b12c0bd4be35 | Saved 4 quadrants
    ✅ Processed: cate9-00034_jpg.rf.83b481150a51a0b186a1f4bb0f1910a6 | Saved 4 quadrants
    ✅ Processed: cate6-00003_jpg.rf.a74ff38dfc403c36db0b4677b6499b09 | Saved 4 quadrants
    ✅ Processed: cate8-00029_jpg.rf.d7de8d30e40ec7f6102668693f1f3fd5 | Saved 4 quadrants
    ✅ Processed: cate6-00141_jpg.rf.2cccae1daf54f19feee6d6293d0a2d83 | Saved 4 quadrants
    ✅ Processed: cate8-00299_jpg.rf.039ed8c0fd89fa56ee421de35d30bd14 | Saved 4 quadrants
    ✅ Processed: cate4-00087_jpg.rf.cc485441335c2f1f9be018eb5133a886 | Saved 4 quadrants
    ✅ Processed: cate8-00030_jpg.rf.844bed08b47e4bf086281d9293b92ca3 | Saved 3 quadrants
    ✅ Processed: cate8-00431_jpg.rf.e9a753767ad65be54775db7af883152b | Saved 3 quadrants
    ✅ Processed: cate6-00051_jpg.rf.117e23e7e710bf1772fd334b0b4b67b4 | Saved 4 quadrants
    ✅ Processed: cate10-00066_jpg.rf.c1d56f5d8a5857b0c54d55c5760fec64 | Saved 2 quadrants
    ✅ Processed: cate3-00009_jpg.rf.8863a6fcb029e6d6766933be7c236cf7 | Saved 4 quadrants
    ✅ Processed: cate5-00060_jpg.rf.aa377e2b8cdd8e7b62b2b2ca9b9b4085 | Saved 2 quadrants
    ✅ Processed: cate4-00111_jpg.rf.1d271e4bcca3671a387db2ec5f0d4341 | Saved 4 quadrants
    ✅ Processed: cate6-00151_jpg.rf.e35e483dfeba488d47f3f3fd704172e8 | Saved 4 quadrants
    ✅ Processed: cate10-00056_jpg.rf.e283da4454bc0607fd574ffbdbb20db6 | Saved 2 quadrants
    ✅ Processed: cate7-00004_jpg.rf.ca37998477725a8349b56db33dffa06e | Saved 4 quadrants
    ✅ Processed: cate6-00093_jpg.rf.7ee6d6ffc4ba11e6e96146f65800cb82 | Saved 4 quadrants
    ✅ Processed: cate7-00002_jpg.rf.eaf80e16fb8ac22309f8365ee3511943 | Saved 4 quadrants



```python
def split_val_data(train_dir, val_dir, val_ratio=0.15):
    train_img_path = Path(train_dir) / "images"
    train_lbl_path = Path(train_dir) / "labels"
    val_img_path = Path(val_dir) / "images"
    val_lbl_path = Path(val_dir) / "labels"

    val_img_path.mkdir(parents=True, exist_ok=True)
    val_lbl_path.mkdir(parents=True, exist_ok=True)

    quadrant_groups = defaultdict(list)
    all_label_files = list(train_lbl_path.glob("*.txt"))

    for lbl_file in all_label_files:
        parts = lbl_file.stem.split('_')
        quad_id = next((p for p in parts if p.startswith('q') and len(p) == 2), None)
        
        if quad_id:
            quadrant_groups[quad_id].append(lbl_file.stem)

    # 3. Perform the move
    print(f"{'Quadrant':<10} | {'Total':<8} | {'Moving to Val':<12}")
    print("-" * 35)

    for quad, stems in quadrant_groups.items():
        random.shuffle(stems)
        num_val = int(len(stems) * val_ratio)
        val_stems = stems[:num_val]

        print(f"{quad:<10} | {len(stems):<8} | {num_val:<12}")

        for stem in val_stems:
            # Handle images (checking for both .jpg and .png)
            img_ext = None
            for ext in ['.jpg', '.png']:
                if (train_img_path / f"{stem}{ext}").exists():
                    img_ext = ext
                    break
            
            if img_ext:
                # Move image
                shutil.move(train_img_path / f"{stem}{img_ext}", 
                            val_img_path / f"{stem}{img_ext}")
                # Move label
                shutil.move(train_lbl_path / f"{stem}.txt", 
                            val_lbl_path / f"{stem}.txt")

    print("\nSplit complete. Validation set is now balanced by quadrant.")
```


```python
split_val_data(
    train_dir="../data/teeth_detection/train",
    val_dir="../data/teeth_detection/val",
    val_ratio=0.15
)
```

    Quadrant   | Total    | Moving to Val
    -----------------------------------
    q2         | 1615     | 242         
    q3         | 1612     | 241         
    q0         | 1588     | 238         
    q1         | 1590     | 238         
    
    Split complete. Validation set is now balanced by quadrant.

