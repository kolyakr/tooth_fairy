```python
import os
import sys

sys.path.append(os.path.abspath("../"))

from pathlib import Path
import json
import numpy as np
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from utils import get_quadrant_crops, apply_clahe
```


```python
TRAIN_IMAGES_SRC = Path("../data/_raw/classification/dentex_data/training_data/quadrant-enumeration-disease/xrays")
TRAIN_LABELS_SRC = Path("../data/_raw/classification/dentex_data/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json")

with open(TRAIN_LABELS_SRC, "r", encoding="utf8") as f:
    data = json.load(f)

data.keys()
```




    dict_keys(['images', 'annotations', 'categories_1', 'categories_2', 'categories_3'])




```python
data["categories_1"]
```




    [{'id': 0, 'name': '1', 'supercategory': '1'},
     {'id': 1, 'name': '2', 'supercategory': '2'},
     {'id': 2, 'name': '3', 'supercategory': '3'},
     {'id': 3, 'name': '4', 'supercategory': '4'}]




```python
data["categories_2"]
```




    [{'id': 0, 'name': '1', 'supercategory': '1'},
     {'id': 1, 'name': '2', 'supercategory': '2'},
     {'id': 2, 'name': '3', 'supercategory': '3'},
     {'id': 3, 'name': '4', 'supercategory': '4'},
     {'id': 4, 'name': '5', 'supercategory': '5'},
     {'id': 5, 'name': '6', 'supercategory': '6'},
     {'id': 6, 'name': '7', 'supercategory': '7'},
     {'id': 7, 'name': '8', 'supercategory': '8'}]




```python
data["categories_3"]
```




    [{'id': 0, 'name': 'Impacted', 'supercategory': 'Impacted'},
     {'id': 1, 'name': 'Caries', 'supercategory': 'Caries'},
     {'id': 2, 'name': 'Periapical Lesion', 'supercategory': 'Periapical Lesion'},
     {'id': 3, 'name': 'Deep Caries', 'supercategory': 'Deep Caries'}]




```python
id_to_name = {item['id']: item['name'] for item in data["categories_3"]}

id_to_name
```




    {0: 'Impacted', 1: 'Caries', 2: 'Periapical Lesion', 3: 'Deep Caries'}




```python
data["images"][0], data["annotations"][0]
```




    ({'height': 1316, 'width': 2744, 'id': 1, 'file_name': 'train_673.png'},
     {'iscrowd': 0,
      'image_id': 1,
      'bbox': [542.0, 698.0, 220.0, 271.0],
      'segmentation': [[621,
        703,
        573,
        744,
        542,
        885,
        580,
        945,
        650,
        969,
        711,
        883,
        762,
        807,
        748,
        741,
        649,
        698]],
      'id': 1,
      'area': 39683,
      'category_id_1': 3,
      'category_id_2': 7,
      'category_id_3': 0})




```python
from collections import defaultdict

# Map image IDs to their filenames
images_dict = {img["id"]: img["file_name"] for img in data["images"]}

# Group segmentation masks, class, and Quadrant ID by image ID
image_to_annotations = defaultdict(list)
for ann in data["annotations"]:
    img_id = ann["image_id"]
    
    if "segmentation" in ann and len(ann["segmentation"]) > 0:
        label_data = {
            "segmentation": ann["segmentation"], 
            "cls": id_to_name[ann["category_id_3"]],
            "quadrant_id": ann["category_id_1"] 
        }
        image_to_annotations[img_id].append(label_data)

all_images = []
all_labels = []

for img_id, file_name in images_dict.items():
    all_images.append(TRAIN_IMAGES_SRC / file_name)
    all_labels.append(image_to_annotations.get(img_id, []))
```


```python
print(f"Total images: {len(all_images)}")
print(f"Total label groups: {len(all_labels)}")

# Count total bounding boxes across all images
count = sum(len(label) for label in all_labels)
print(f"Total segm. masks: {count}")
```

    Total images: 705
    Total label groups: 705
    Total segm. masks: 3529



```python
import matplotlib.gridspec as gridspec

def draw_segmentations(img, annotations, color_map, offset=(0, 0), alpha=0.4):
    """Draws alpha-blended ground truth masks with coordinate offsets."""
    overlay = img.copy()
    mask_layer = img.copy()
    ox, oy = offset
    
    for ann in annotations:
        cls_name = ann['cls']
        color = color_map.get(cls_name, (255, 255, 255))
        
        for seg in ann['segmentation']:
            # Apply the YOLO crop offset to the JSON coordinates
            pts = np.array(seg, np.int32).reshape((-1, 1, 2))
            pts = pts - np.array([ox, oy]) 
            
            cv2.fillPoly(mask_layer, [pts], color)
            cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)
            
            x, y, w, h = cv2.boundingRect(pts)
            
            # Only draw text if the mask is within this specific cropped image
            if x >= 0 and y >= 0 and x < img.shape[1] and y < img.shape[0]:
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(cls_name, font, 0.6, 2)
                cv2.rectangle(overlay, (x, max(0, y - th - 5)), (x + tw, max(0, y)), color, -1)
                cv2.putText(overlay, cls_name, (x, max(0, y - 2)), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return cv2.addWeighted(mask_layer, alpha, overlay, 1 - alpha, 0)
```


```python
def visualize_filtered_pipeline(image_path, labels, class_names):
    # 1. Extract YOLO quadrants
    orig_img, crops_data, filename = get_quadrant_crops(image_path, padding=40)
    if not crops_data: return

    color_map = {name: tuple(map(int, cv2.cvtColor(np.uint8([[[int(180 * i / len(class_names)), 200, 255]]]), cv2.COLOR_HSV2BGR)[0][0])) 
                 for i, name in enumerate(class_names)}
    quad_colors = {0: (0, 255, 0), 1: (0, 165, 255), 2: (255, 150, 0), 3: (255, 0, 255)}

    num_quads = len(crops_data)
    fig, axes = plt.subplots(nrows=2 + num_quads, ncols=1, figsize=(18, 10 * (2 + num_quads)))

    # --- Panel 1: Original ---
    axes[0].imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"1. Original OPG: {filename}", fontsize=20, fontweight='bold'); axes[0].axis('off')

    # --- Panel 2: Full Map with Quadrant Overlays ---
    full_annotated = orig_img.copy()
    quad_layer = orig_img.copy()
    
    # First: Draw YOLO Quadrant Masks
    for item in crops_data:
        if item["mask_xy"] is not None:
            pts = np.array(item["mask_xy"], dtype=np.int32)
            color = quad_colors.get(item["class_id"], (255, 255, 255))
            cv2.fillPoly(quad_layer, [pts], color)
            cv2.polylines(full_annotated, [pts], True, color, 4) # Thick boundary
    
    # Blend quadrants first (alpha 0.25)
    full_annotated = cv2.addWeighted(quad_layer, 0.25, full_annotated, 0.75, 0)
    # Second: Draw Teeth/Disease segmentations on top
    full_annotated = draw_segmentations(full_annotated, labels, color_map, offset=(0, 0), alpha=0.5)
    
    axes[1].imshow(cv2.cvtColor(full_annotated, cv2.COLOR_BGR2RGB))
    axes[1].set_title("2. Full Diagnostic Map (Quadrants + Diseases)", fontsize=20, fontweight='bold'); axes[1].axis('off')

    # --- Rows 3+: Filtered Quadrants ---
    sorted_crops = sorted(crops_data, key=lambda x: x["class_id"])
    for i, item in enumerate(sorted_crops):
        q_id, crop_img, offset = item["class_id"], item["crop"], item["top_left"]
        filtered_labels = [l for l in labels if l["quadrant_id"] == q_id]
        
        # Color the crop background slightly with its quadrant color
        q_color = quad_colors.get(q_id, (255, 255, 255))
        colored_crop = np.full_like(crop_img, q_color)
        crop_base = cv2.addWeighted(colored_crop, 0.1, crop_img, 0.9, 0)
        
        annotated_crop = draw_segmentations(crop_base, filtered_labels, color_map, offset=offset)
        axes[i+2].imshow(cv2.cvtColor(annotated_crop, cv2.COLOR_BGR2RGB))
        axes[i+2].set_title(f"3.{i+1} Quadrant {q_id} (Filtered)", fontsize=18, color='darkblue', fontweight='bold'); axes[i+2].axis('off')

    plt.tight_layout(); plt.show()
```


```python
NUM_TO_VISUALIZE = 1
class_names = list(id_to_name.values())

for i, (image_path, labels) in enumerate(zip(all_images[:NUM_TO_VISUALIZE], all_labels[:NUM_TO_VISUALIZE])):
    visualize_filtered_pipeline(image_path, labels, class_names)
```


    
![png](notebooks/05_tooth_classification_dataset_files/notebooks/05_tooth_classification_dataset_11_0.png)
    



```python
import os
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
NEW_DATASET_ROOT = Path("../data/yolo_quadrant_dataset")
IMAGE_OUT = NEW_DATASET_ROOT / "train/images"
LABEL_OUT = NEW_DATASET_ROOT / "train/labels"

# Ensure directories exist
IMAGE_OUT.mkdir(parents=True, exist_ok=True)
LABEL_OUT.mkdir(parents=True, exist_ok=True)

# Blacklist as requested
BLACKLIST = {"train_516.png"}

# New mapping for YOLO (0-indexed)
disease_name_to_id = {name: i for i, name in id_to_name.items()}
```


```python
def process_quadrant_dataset(all_images, all_labels):
    processed_count = 0
    background_count = 0
    
    for img_path, annotations in tqdm(zip(all_images, all_labels), total=len(all_images)):
        
        # 1. Check Blacklist (e.g., train_516.png)
        if img_path.name in BLACKLIST:
            continue
            
        # 2. Extract Quadrants using YOLO
        orig_img, crops_data, filename = get_quadrant_crops(img_path, padding=40)
        
        if not crops_data:
            continue

        for item in crops_data:
            q_id = item["class_id"]
            crop_img = item["crop"]
            crop_h, crop_w = crop_img.shape[:2]
            ox, oy = item["top_left"] 

            local_ann_lines = []
            quad_annotations = [a for a in annotations if a["quadrant_id"] == q_id]

            for ann in quad_annotations:
                cls_id = disease_name_to_id[ann["cls"]]
                
                for seg in ann["segmentation"]:
                    # Cast to float for math operations
                    coords = np.array(seg, dtype=np.float32).reshape(-1, 2)
                    
                    # Coordinate shift relative to the crop
                    coords[:, 0] -= ox
                    coords[:, 1] -= oy
                    
                    # Filter: If the entire mask is outside the crop bounds, discard it
                    if np.all(coords[:, 0] < 0) or np.all(coords[:, 0] > crop_w) or \
                       np.all(coords[:, 1] < 0) or np.all(coords[:, 1] > crop_h):
                        continue

                    # Normalize by crop size
                    coords[:, 0] /= crop_w
                    coords[:, 1] /= crop_h
                    
                    # Boundary Clipping: "Crop" the mask to the image edges [0.0, 1.0]
                    coords = np.clip(coords, 0, 1)
                    
                    # Format for YOLO: 'cls x1 y1 x2 y2 ...'
                    flattened_coords = coords.flatten()
                    line = f"{cls_id} " + " ".join([f"{c:.6f}" for c in flattened_coords])
                    local_ann_lines.append(line)

            # 3. Save Logic
            new_filename = f"{img_path.stem}_q{q_id}.png"
            label_filename = f"{img_path.stem}_q{q_id}.txt"
            
            # ALWAYS save the image (even without diseases)
            cv2.imwrite(str(IMAGE_OUT / new_filename), crop_img)
            
            # ALWAYS save the label file (standard YOLO practice for background images)
            with open(LABEL_OUT / label_filename, "w") as f:
                if local_ann_lines:
                    f.write("\n".join(local_ann_lines))
                    processed_count += 1
                else:
                    # Empty file = Background image
                    f.write("") 
                    background_count += 1

    print(f"\nProcessing Complete:")
    print(f"- Images with Diseases: {processed_count}")
    print(f"- Background Images (No Disease): {background_count}")
    return processed_count + background_count

# Run the processing
total_saved = process_quadrant_dataset(all_images, all_labels)
```

    100%|██████████| 705/705 [04:39<00:00,  2.52it/s]

    
    Processing Complete:
    - Images with Diseases: 1910
    - Background Images (No Disease): 905


    



```python
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def visualize_class_distribution(label_path, id_to_name_map):
    # 1. Collect all class IDs and count empty (background) files
    all_class_ids = []
    background_count = 0
    label_files = list(label_path.glob("*.txt"))
    
    for lb_file in label_files:
        # Check if the file is empty (background image)
        if lb_file.stat().st_size == 0:
            background_count += 1
        else:
            with open(lb_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        all_class_ids.append(int(parts[0]))
    
    # 2. Count occurrences and map to names
    counts = Counter(all_class_ids)
    
    # Map class IDs to names and add the Background category
    data_for_plot = {id_to_name_map[i]: counts.get(i, 0) for i in id_to_name_map.keys()}
    data_for_plot["Background (Empty)"] = background_count
    
    # Sort by count for better visualization
    sorted_data = dict(sorted(data_for_plot.items(), key=lambda item: item[1], reverse=True))
    
    # 3. Plotting
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    # Using a distinct color palette
    colors = sns.color_palette("magma", len(sorted_data))
    bars = plt.bar(sorted_data.keys(), sorted_data.values(), color=colors)
    
    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), 
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.title("Detailed Class Distribution (Including Background Crops)", fontsize=16, fontweight='bold')
    plt.xlabel("Category", fontsize=13)
    plt.ylabel("Total Instances / Images", fontsize=13)
    plt.xticks(rotation=45, ha='right')
    
    # Highlight the Background bar if it exists
    for i, label in enumerate(sorted_data.keys()):
        if label == "Background (Empty)":
            bars[i].set_edgecolor('red')
            bars[i].set_linewidth(2)

    plt.tight_layout()
    plt.show()

    # 4. Print Summary Table
    total_samples = len(label_files)
    print(f"{'Category Name':<25} | {'Count':<10} | {'Percentage':<10}")
    print("-" * 50)
    for name, count in sorted_data.items():
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"{name:<25} | {count:<10} | {percentage:>8.2f}%")
    print("-" * 50)
    print(f"{'Total Files Scanned':<25} | {total_samples:<10}")

# Run the updated visualization
visualize_class_distribution(LABEL_OUT, id_to_name)
```


    
![png](notebooks/05_tooth_classification_dataset_files/notebooks/05_tooth_classification_dataset_14_0.png)
    


    Category Name             | Count      | Percentage
    --------------------------------------------------
    Caries                    | 2186       |    77.66%
    Background (Empty)        | 905        |    32.15%
    Impacted                  | 604        |    21.46%
    Deep Caries               | 577        |    20.50%
    Periapical Lesion         | 158        |     5.61%
    --------------------------------------------------
    Total Files Scanned       | 2815      



```python
import os
import random
from pathlib import Path

def prune_background_images(image_dir, label_dir, target_ratio=0.10):
    all_label_files = list(label_dir.glob("*.txt"))
    
    # 1. Separate backgrounds from annotated files
    background_files = []
    annotated_files = []
    
    for lb_file in all_label_files:
        if lb_file.stat().st_size == 0:
            background_files.append(lb_file)
        else:
            annotated_files.append(lb_file)
            
    total_annotated = len(annotated_files)
    current_bg_count = len(background_files)
    
    print(f"Current Annotated Files: {total_annotated}")
    print(f"Current Background Files: {current_bg_count}")
    
    # 2. Calculate the target number of background files
    # Math: If we want backgrounds to be 10% of TOTAL, then annotated is 90%
    # Target_BG = (Total_Annotated / 0.90) * 0.10
    target_bg_count = int((total_annotated / (1 - target_ratio)) * target_ratio)
    
    files_to_delete_count = current_bg_count - target_bg_count
    
    if files_to_delete_count <= 0:
        print("Background ratio is already at or below target. No pruning needed.")
        return
        
    print(f"Target Background Count ({int(target_ratio*100)}%): ~{target_bg_count}")
    print(f"Pruning {files_to_delete_count} empty images...")
    
    # 3. Randomly select background files to delete
    # We use a seed for reproducibility
    random.seed(42)
    files_to_delete = random.sample(background_files, files_to_delete_count)
    
    deleted_count = 0
    for lb_file in files_to_delete:
        # Get the corresponding image file
        img_file = image_dir / f"{lb_file.stem}.png"
        
        # Delete both
        if lb_file.exists():
            os.remove(lb_file)
        if img_file.exists():
            os.remove(img_file)
            
        deleted_count += 1
        
    print(f"Successfully deleted {deleted_count} background samples.")

# Run the pruning script
prune_background_images(IMAGE_OUT, LABEL_OUT, target_ratio=0.20)

# Re-run the visualizer to confirm!
print("\n--- New Dataset Distribution ---")
visualize_class_distribution(LABEL_OUT, id_to_name)
```

    Current Annotated Files: 1910
    Current Background Files: 905
    Target Background Count (20%): ~477
    Pruning 428 empty images...
    Successfully deleted 428 background samples.
    
    --- New Dataset Distribution ---



    
![png](notebooks/05_tooth_classification_dataset_files/notebooks/05_tooth_classification_dataset_15_1.png)
    


    Category Name             | Count      | Percentage
    --------------------------------------------------
    Caries                    | 2186       |    91.58%
    Impacted                  | 604        |    25.30%
    Deep Caries               | 577        |    24.17%
    Background (Empty)        | 477        |    19.98%
    Periapical Lesion         | 158        |     6.62%
    --------------------------------------------------
    Total Files Scanned       | 2387      



```python
import os
import shutil
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def stratified_split_yolo(root_path, image_subfolder="train/images", label_subfolder="train/labels", val_size=0.2):
    image_dir = root_path / image_subfolder
    label_dir = root_path / label_subfolder
    
    # 1. Map files to their "Rarest Class" for stratification
    # Order of importance (rarest to most common)
    class_priority = [0, 2, 1, 3] # Adjusted based on your counts (e.g., 0=Lesion, 2=Deep Caries...)
    
    all_images = sorted(list(image_dir.glob("*.png")))
    stratify_labels = []
    
    for img_p in all_images:
        lb_p = label_dir / f"{img_p.stem}.txt"
        
        if lb_p.exists() and lb_p.stat().st_size > 0:
            with open(lb_p, "r") as f:
                classes_in_file = [int(line.split()[0]) for line in f.readlines()]
            
            # Assign the rarest class found in the file as the stratification anchor
            found_priority = [c for c in class_priority if c in classes_in_file]
            stratify_labels.append(found_priority[0] if found_priority else classes_in_file[0])
        else:
            stratify_labels.append(-1) # Background
            
    # 2. Perform the Stratified Split
    train_imgs, val_imgs = train_test_split(
        all_images, 
        test_size=val_size, 
        stratify=stratify_labels, 
        random_state=42
    )

    # 3. Create Directories
    val_img_path = root_path / "val/images"
    val_lb_path = root_path / "val/labels"
    val_img_path.mkdir(parents=True, exist_ok=True)
    val_lb_path.mkdir(parents=True, exist_ok=True)

    # 4. Move Files
    print(f"Moving {len(val_imgs)} samples to validation set...")
    for img_p in val_imgs:
        # Move Image
        shutil.move(str(img_p), str(val_img_path / img_p.name))
        
        # Move Label
        lb_p = label_dir / f"{img_p.stem}.txt"
        if lb_p.exists():
            shutil.move(str(lb_p), str(val_lb_path / lb_p.name))

    print(f"Split Complete! Train: {len(train_imgs)} | Val: {len(val_imgs)}")

# Run the split on your new dataset root
stratified_split_yolo(NEW_DATASET_ROOT)
```

    Moving 478 samples to validation set...
    Split Complete! Train: 1909 | Val: 478



```python
def check_split_proportions(root_path, id_to_name):
    splits = ['train', 'val']
    results = {}

    for split in splits:
        counts = {name: 0 for name in id_to_name.values()}
        labels = list((root_path / split / "labels").glob("*.txt"))
        
        for lb in labels:
            with open(lb, "r") as f:
                for line in f:
                    cls_id = int(line.split()[0])
                    counts[id_to_name[cls_id]] += 1
        results[split] = counts

    # Print Comparison Table
    print(f"{'Pathology':<20} | {'Train %':<10} | {'Val %':<10}")
    print("-" * 45)
    for name in id_to_name.values():
        t_count = results['train'][name]
        v_count = results['val'][name]
        total = t_count + v_count
        t_perc = (t_count / total * 100) if total > 0 else 0
        v_perc = (v_count / total * 100) if total > 0 else 0
        print(f"{name:<20} | {t_perc:>8.1f}% | {v_perc:>8.1f}%")

check_split_proportions(NEW_DATASET_ROOT, id_to_name)
```

    Pathology            | Train %    | Val %     
    ---------------------------------------------
    Impacted             |     80.0% |     20.0%
    Caries               |     79.6% |     20.4%
    Periapical Lesion    |     77.2% |     22.8%
    Deep Caries          |     83.4% |     16.6%



```python
val_label_dir = Path('../data/yolo_quadrant_dataset/val/labels')
counts = Counter()

for lb in val_label_dir.glob("*.txt"):
    with open(lb, 'r') as f:
        for line in f:
            counts[int(line.split()[0])] += 1

print("--- Actual IDs found in your .txt files ---")
for cls_id, count in sorted(counts.items()):
    print(f"ID {cls_id}: {count} instances")
```

    --- Actual IDs found in your .txt files ---
    ID 0: 121 instances
    ID 1: 447 instances
    ID 2: 36 instances
    ID 3: 96 instances



```python
from pathlib import Path
from tqdm import tqdm

def refine_yolo_dataset(root_path):
    id_map = {0: 0, 1: 1, 3: 2}
    
    splits = ['train', 'val']
    
    for split in splits:
        label_dir = Path(root_path) / split / "labels"
        label_files = list(label_dir.glob("*.txt"))
        
        print(f"Filtering {split} labels...")
        
        for lb_path in tqdm(label_files):
            new_lines = []
            
            with open(lb_path, "r") as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.split()
                if not parts: continue
                
                cls_id = int(parts[0])
                
                # Only keep and remap if the ID is in our map
                if cls_id in id_map:
                    new_id = id_map[cls_id]
                    # Reconstruct the line with the new class ID
                    new_line = f"{new_id} {' '.join(parts[1:])}\n"
                    new_lines.append(new_line)
            
            # Overwrite the label file with filtered/remapped content
            with open(lb_path, "w") as f:
                f.writelines(new_lines)

# Execute
DATASET_ROOT = Path("../data/yolo_quadrant_dataset")
refine_yolo_dataset(DATASET_ROOT)

# Updated mapping for your check function
new_id_to_name = {
    0: "Impacted",
    1: "Caries",
    2: "Deep Caries"
}

print("\n✅ Dataset refined! Running sanity check...")
check_split_proportions(DATASET_ROOT, new_id_to_name)
```

    Filtering train labels...


    100%|██████████| 1909/1909 [00:00<00:00, 9670.28it/s]

    Filtering val labels...

    


    


    100%|██████████| 478/478 [00:00<00:00, 9914.09it/s]


    
    ✅ Dataset refined! Running sanity check...
    Pathology            | Train %    | Val %     
    ---------------------------------------------
    Impacted             |     80.0% |     20.0%
    Caries               |     79.6% |     20.4%
    Deep Caries          |     83.4% |     16.6%



```python
LABEL_OUT = NEW_DATASET_ROOT / "train/labels"
visualize_class_distribution(LABEL_OUT, new_id_to_name)
```


    
![png](notebooks/05_tooth_classification_dataset_files/notebooks/05_tooth_classification_dataset_20_0.png)
    


    Category Name             | Count      | Percentage
    --------------------------------------------------
    Caries                    | 1739       |    91.09%
    Impacted                  | 483        |    25.30%
    Deep Caries               | 481        |    25.20%
    Background (Empty)        | 396        |    20.74%
    --------------------------------------------------
    Total Files Scanned       | 1909      



```python
LABEL_OUT = NEW_DATASET_ROOT / "val/labels"
visualize_class_distribution(LABEL_OUT, new_id_to_name)
```


    
![png](notebooks/05_tooth_classification_dataset_files/notebooks/05_tooth_classification_dataset_21_0.png)
    


    Category Name             | Count      | Percentage
    --------------------------------------------------
    Caries                    | 447        |    93.51%
    Impacted                  | 121        |    25.31%
    Background (Empty)        | 102        |    21.34%
    Deep Caries               | 96         |    20.08%
    --------------------------------------------------
    Total Files Scanned       | 478       

