```python
import os
import sys
import random
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# Ensure utils is accessible
sys.path.append(os.path.abspath("../"))
from utils import get_quadrant_crops

# --- Configuration ---
orig_images_dir = Path("../data/_raw/_reserve/Panoramic radiographs with periapical lesions Dataset/Periapical Dataset/Periapical Lesions/Original JPG Images")
annots_dir = Path("../data/_raw/_reserve/Panoramic radiographs with periapical lesions Dataset/Periapical Dataset/Periapical Lesions/Image Annots")

QUAD_NAMES = {0: "Upper Right", 1: "Upper Left", 2: "Lower Left", 3: "Lower Right"}
QUAD_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)] 
LESION_COLOR = (0, 255, 255) # Bright Yellow
AREA_THRESHOLD = 0.3  # Only keep boxes if at least 30% is visible in the crop

def load_lesion_bboxes(xml_path):
    bboxes = []
    if not xml_path.exists(): return bboxes
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        bboxes.append([
            float(box.find("xmin").text), float(box.find("ymin").text),
            float(box.find("xmax").text), float(box.find("ymax").text)
        ])
    return bboxes

def visualize_random_samples(num_samples=100):
    all_image_paths = list(orig_images_dir.glob("*.jpg"))
    
    # Randomly sample images to get a diverse view of the dataset
    if len(all_image_paths) < num_samples:
        num_samples = len(all_image_paths)
    
    sampled_paths = random.sample(all_image_paths, num_samples)
    
    for idx, img_path in enumerate(sampled_paths):
        print(f"[{idx+1}/{num_samples}] Processing: {img_path.name}")
        
        # 1. Get crops and global annotations
        # Using padding=200 as requested to better capture midline lesions
        orig_img, crops_data, filename = get_quadrant_crops(img_path)
        global_lesions = load_lesion_bboxes(annots_dir / f"{img_path.stem}.xml")
        
        fig, axes = plt.subplots(5, 1, figsize=(15, 45))
        
        # --- ROW 0: Full Image with Masks and ALL Global Lesions ---
        full_vis = orig_img.copy()
        mask_overlay = np.zeros_like(orig_img)
        
        for i, data in enumerate(crops_data):
            if data["mask_xy"] is not None:
                pts = np.array(data["mask_xy"], dtype=np.int32)
                cv2.fillPoly(mask_overlay, [pts], QUAD_COLORS[i % len(QUAD_COLORS)])
                cv2.polylines(full_vis, [pts], True, (255, 255, 255), 5)

        cv2.addWeighted(mask_overlay, 0.3, full_vis, 0.7, 0, full_vis)
        
        for l_box in global_lesions:
            lx1, ly1, lx2, ly2 = map(int, l_box)
            cv2.rectangle(full_vis, (lx1, ly1), (lx2, ly2), LESION_COLOR, 12)
            
        axes[0].imshow(cv2.cvtColor(full_vis, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"FULL OPG: {filename}\n(Inclusive Mode + {int(AREA_THRESHOLD*100)}% Area Filter)", fontsize=20, fontweight='bold')
        axes[0].axis("off")
        
        # --- ROWS 1-4: Individual Crops with Filtered Lesion Mapping ---
        for q_idx, q_data in enumerate(crops_data):
            if q_idx >= 4: break 
            
            crop_img = q_data["crop"].copy()
            px1, py1 = q_data["top_left"]
            crop_h, crop_w = crop_img.shape[:2]
            
            for l_box in global_lesions:
                lx1, ly1, lx2, ly2 = l_box
                
                # Intersection check
                if (lx2 > px1 and lx1 < px1 + crop_w and 
                    ly2 > py1 and ly1 < py1 + crop_h):
                    
                    # Area Retention Logic
                    orig_area = (lx2 - lx1) * (ly2 - ly1)
                    
                    # Calculate clipped boundaries
                    c_x1, c_y1 = max(px1, lx1), max(py1, ly1)
                    c_x2, c_y2 = min(px1 + crop_w, lx2), min(py1 + crop_h, ly2)
                    clipped_area = (c_x2 - c_x1) * (c_y2 - c_y1)
                    
                    # Only visualize/label if enough of the lesion is visible
                    if (clipped_area / orig_area) >= AREA_THRESHOLD:
                        local_x1 = int(c_x1 - px1)
                        local_y1 = int(c_y1 - py1)
                        local_x2 = int(c_x2 - px1)
                        local_y2 = int(c_y2 - py1)
                        
                        cv2.rectangle(crop_img, (local_x1, local_y1), (local_x2, local_y2), LESION_COLOR, 12)
            
            axes[q_idx+1].imshow(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            axes[q_idx+1].set_title(f"QUADRANT: {QUAD_NAMES[q_idx]}", fontsize=18, color='blue')
            axes[q_idx+1].axis("off")
            
        plt.subplots_adjust(hspace=0.2)
        plt.show()

# Execute 100 random visualizations
visualize_random_samples(num_samples=1)
```

    [1/1] Processing: 12261.jpg



    
![png](notebooks/06_periapical_files/notebooks/06_periapical_0_1.png)
    



```python
import os
import random
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# --- Configuration ---
output_root = Path("../data/periapical_cropped_detection") 
train_ratio = 0.8 
AREA_THRESHOLD = 0.3 # Label threshold
BG_TO_POS_RATIO = 1.0  
PADDING = 60  

def convert_to_yolo(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    x_center, y_center = x1 + (width / 2), y1 + (height / 2)
    return f"0 {x_center/img_w:.6f} {y_center/img_h:.6f} {width/img_w:.6f} {height/img_h:.6f}"

def process_dataset():
    for split in ['train', 'val']:
        (output_root / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_root / split / 'labels').mkdir(parents=True, exist_ok=True)

    all_image_paths = list(orig_images_dir.glob("*.jpg"))
    random.shuffle(all_image_paths)
    
    split_idx = int(len(all_image_paths) * train_ratio)
    splits = [('train', all_image_paths[:split_idx]), ('val', all_image_paths[split_idx:])]

    for split_name, image_list in splits:
        print(f"\n📦 Analyzing {split_name} split...")
        pos_metadata = []
        neg_metadata = []
        excluded_count = 0

        for img_path in tqdm(image_list, desc="Scanning Annotations"):
            global_lesions = load_lesion_bboxes(annots_dir / f"{img_path.stem}.xml")
            _, crops_data, _ = get_quadrant_crops(img_path, padding=PADDING)
            
            if len(crops_data) != 4:
                continue

            for q_idx, q_data in enumerate(crops_data):
                px1, py1 = q_data["top_left"]
                h_crop, w_crop = q_data["crop"].shape[:2]
                
                yolo_labels = []
                is_dirty_background = False # Flag for "slivers"

                for l_box in global_lesions:
                    lx1, ly1, lx2, ly2 = l_box
                    
                    # Calculate visibility ratio
                    if (lx2 > px1 and lx1 < px1 + w_crop and ly2 > py1 and ly1 < py1 + h_crop):
                        orig_area = (lx2 - lx1) * (ly2 - ly1)
                        c_x1, c_y1 = max(px1, lx1), max(py1, ly1)
                        c_x2, c_y2 = min(px1 + w_crop, lx2), min(py1 + h_crop, ly2)
                        clipped_area = (c_x2 - c_x1) * (c_y2 - c_y1)
                        visibility = clipped_area / orig_area

                        if visibility >= AREA_THRESHOLD:
                            # Valid Positive Label
                            local_box = [c_x1 - px1, c_y1 - py1, c_x2 - px1, c_y2 - py1]
                            yolo_labels.append(convert_to_yolo(local_box, w_crop, h_crop))
                        elif visibility > 0:
                            # It's a sliver! We cannot use this as a clean background
                            is_dirty_background = True

                meta = (img_path, q_idx, yolo_labels)
                
                if yolo_labels:
                    # If it has at least one valid lesion, it's positive 
                    # (even if it ALSO has a sliver from another tooth)
                    pos_metadata.append(meta)
                elif is_dirty_background:
                    # No valid lesions, but has an unannotated sliver -> DISCARD
                    excluded_count += 1
                else:
                    # Pure, clean background
                    neg_metadata.append(meta)

        print(f"🗑️ Discarded {excluded_count} 'dirty' quadrants containing unlabeled slivers.")

        # --- Balanced Selection ---
        num_neg_to_keep = int(len(pos_metadata) * BG_TO_POS_RATIO)
        selected_negs = random.sample(neg_metadata, min(len(neg_metadata), num_neg_to_keep))
        final_list = pos_metadata + selected_negs
        random.shuffle(final_list)

        # --- Second Pass: Grouped Disk Writes ---
        tasks = defaultdict(list)
        for img_p, q_i, labs in final_list:
            tasks[img_p].append((q_i, labs))

        print(f"💾 Saving {len(final_list)} images...")
        for img_path, q_tasks in tqdm(tasks.items(), desc="Writing Images"):
            orig_img, crops_data, _ = get_quadrant_crops(img_path, padding=PADDING)
            for q_idx, labels in q_tasks:
                crop_img = crops_data[q_idx]["crop"]
                base_name = f"{img_path.stem}_q{q_idx}"
                cv2.imwrite(str(output_root / split_name / 'images' / f"{base_name}.jpg"), crop_img)
                with open(output_root / split_name / 'labels' / f"{base_name}.txt", "w") as f:
                    f.write("\n".join(labels))

    print(f"\n✅ Balanced and Cleaned Dataset Ready at {output_root.absolute()}")

process_dataset()
```

    
    📦 Analyzing train split...


    Scanning Annotations: 100%|██████████| 3139/3139 [14:55<00:00,  3.50it/s]


    🗑️ Discarded 365 'dirty' quadrants containing unlabeled slivers.
    💾 Saving 9770 images...


    Writing Images: 100%|██████████| 3139/3139 [15:13<00:00,  3.43it/s]


    
    📦 Analyzing val split...


    Scanning Annotations: 100%|██████████| 785/785 [03:43<00:00,  3.52it/s]


    🗑️ Discarded 88 'dirty' quadrants containing unlabeled slivers.
    💾 Saving 2472 images...


    Writing Images: 100%|██████████| 785/785 [03:47<00:00,  3.45it/s]

    
    ✅ Balanced and Cleaned Dataset Ready at /Users/nickol/My Fucking Stuff/machine learning/code/tooth_fairy/notebooks/../data/periapical_cropped_detection


    



```python
import os
from pathlib import Path
from collections import Counter
import pandas as pd

# --- Configuration ---
# Point to the root directory created in the previous step
dataset_root = Path("../data/periapical_cropped_detection") 
QUAD_NAMES = {0: "Upper Right", 1: "Upper Left", 2: "Lower Left", 3: "Lower Right"}

def generate_dataset_report(root_path):
    stats = []

    for split in ['train', 'val']:
        img_dir = root_path / split / 'images'
        lbl_dir = root_path / split / 'labels'
        
        images = list(img_dir.glob("*.jpg"))
        
        split_data = {
            "Split": split,
            "Total Images": len(images),
            "Background (Empty)": 0,
            "Positive (With Lesions)": 0,
            "Total Bboxes": 0,
            "Q0_UR": 0, "Q1_UL": 0, "Q2_LL": 0, "Q3_LR": 0
        }

        for img_path in images:
            # 1. Identify Quadrant from filename suffix (_q0, _q1, etc.)
            q_idx = int(img_path.stem.split('_q')[-1])
            split_data[f"Q{q_idx}_{['UR','UL','LL','LR'][q_idx]}"] += 1
            
            # 2. Check Labels
            label_path = lbl_dir / f"{img_path.stem}.txt"
            if label_path.exists() and os.path.getsize(label_path) > 0:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    split_data["Positive (With Lesions)"] += 1
                    split_data["Total Bboxes"] += len(lines)
            else:
                split_data["Background (Empty)"] += 1
        
        stats.append(split_data)

    # Display Results
    df = pd.DataFrame(stats)
    
    print("-" * 30)
    print("📊 PANODENT AI: DATASET REPORT")
    print("-" * 30)
    print(df.to_string(index=False))
    
    # Validation Checks
    print("\n🔍 SANITY CHECKS:")
    for split in stats:
        # Verify quadrant balance
        q_counts = [split[f"Q{i}_{s}"] for i, s in enumerate(['UR','UL','LL','LR'])]
        if len(set(q_counts)) == 1:
            print(f"✅ {split['Split']}: Quadrants are perfectly stratified ({q_counts[0]} each).")
        else:
            print(f"⚠️ {split['Split']}: Quadrant imbalance detected! {q_counts}")
            
    return df

# Run the report
report_df = generate_dataset_report(dataset_root)
```

    ------------------------------
    📊 PANODENT AI: DATASET REPORT
    ------------------------------
    Split  Total Images  Background (Empty)  Positive (With Lesions)  Total Bboxes  Q0_UR  Q1_UL  Q2_LL  Q3_LR
    train          9770                4885                     4885          6368   2371   2393   2477   2529
      val          2472                1236                     1236          1631    587    613    630    642
    
    🔍 SANITY CHECKS:
    ⚠️ train: Quadrant imbalance detected! [2371, 2393, 2477, 2529]
    ⚠️ val: Quadrant imbalance detected! [587, 613, 630, 642]



```python

```
