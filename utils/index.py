from ultralytics import YOLO
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def segment_quadrants(image_path, conf=0.5):
    clahe_img = apply_clahe(image_path)

    clahe_bgr = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)

    quadrant_seg_model_path = "../models/quadrant segmentation/best.pt"
    model = YOLO(quadrant_seg_model_path)

    results = model.predict(
        source=clahe_bgr,  
        conf=conf,
        verbose=False
    )

    result = results[0]
    
    keep_indices = []
    for cls in [0, 1, 2, 3]:
        indices = torch.where(result.boxes.cls == cls)[0]
        if len(indices) > 0:
            class_confs = result.boxes.conf[indices]
            best_idx_relative = torch.argmax(class_confs)
            best_idx = indices[best_idx_relative].item() 
            keep_indices.append(best_idx)
            
    filtered_result = result[keep_indices]
    
    return filtered_result

def apply_clahe(src_img_path):
    img = cv2.imread(str(src_img_path), cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl_img = clahe.apply(img)

    return cl_img

def get_quadrant_crops(image_path, padding=40):
    result = segment_quadrants(image_path, conf=0.01) 
    orig_img = result.orig_img.copy()
    h_img, w_img = orig_img.shape[:2]
    filename = getattr(image_path, 'name', str(image_path))

    if result.boxes is None or len(result.boxes) == 0:
        return orig_img, [], filename

    boxes = result.boxes.xyxy.cpu().numpy().astype(np.int32)
    classes = result.boxes.cls.cpu().numpy().astype(np.int32)
    masks_xy = result.masks.xy if result.masks is not None else [None] * len(boxes)
    
    crops_data = []
    for box, cls_id, m_xy in zip(boxes, classes, masks_xy):
        x1, y1, x2, y2 = box
        px1, py1 = max(0, x1 - padding), max(0, y1 - padding)
        px2, py2 = min(w_img, x2 + padding), min(h_img, y2 + padding)
        
        crop = orig_img[py1:py2, px1:px2].copy()
        
        crops_data.append({
            "crop": crop,
            "class_id": int(cls_id),
            "mask_xy": m_xy, 
            "original_box": [x1, y1, x2, y2],
            "top_left": (px1, py1) 
        })
        
    return orig_img, crops_data, filename

def visualize_quadrant_crops(processed_image, crops_data, filename, alpha=0.3):
    if not crops_data:
        plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Analysis Failed: {filename}")
        plt.axis('off'); plt.show()
        return

    fig = plt.figure(figsize=(16, 18))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 1, 1])
    quad_colors = [(0, 255, 0), (0, 165, 255), (255, 150, 0), (255, 0, 255)]
    
    overlay_pano = processed_image.copy()
    for item in crops_data:
        cls_id = item["class_id"]
        color = quad_colors[cls_id] if cls_id < 4 else (255, 255, 255)
        
        # Draw Mask
        if item["mask_xy"] is not None:
            pts = np.array(item["mask_xy"], dtype=np.int32)
            cv2.fillPoly(overlay_pano, [pts], color)
        
        x1, y1, x2, y2 = item["original_box"]
        cv2.rectangle(overlay_pano, (x1, y1), (x2, y2), color, 3)
        cv2.putText(overlay_pano, f"Q{cls_id}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    combined_pano = cv2.addWeighted(overlay_pano, alpha, processed_image, 1 - alpha, 0)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(cv2.cvtColor(combined_pano, cv2.COLOR_BGR2RGB))
    ax0.set_title(f"{filename}", fontsize=20, fontweight='bold')
    ax0.axis('off')

    grid_pos = {0: (1, 0), 1: (1, 1), 2: (2, 0), 3: (2, 1)}
    for item in crops_data:
        cls_id = item["class_id"]
        if cls_id not in grid_pos: continue 
        
        crop = item["crop"].copy()
        px1, py1 = item["top_left"]
        color = quad_colors[cls_id]
        
        if item["mask_xy"] is not None:
            crop_overlay = crop.copy()
            pts_local = np.array(item["mask_xy"], dtype=np.int32)
            pts_local[:, 0] -= px1
            pts_local[:, 1] -= py1
            
            cv2.fillPoly(crop_overlay, [pts_local], color)
            crop = cv2.addWeighted(crop_overlay, alpha, crop, 1 - alpha, 0)

        row, col = grid_pos[cls_id]
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Quadrant {cls_id}", fontsize=14, color='darkblue')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def visualize_teeth(result, alpha=0.35):
    orig_img = result.orig_img.copy()
    overlay = orig_img.copy()
    
    if result.masks is None:
        plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        return

    masks = result.masks.xy
    boxes = result.boxes.xyxy.cpu().numpy().astype(np.int32)
    classes = result.boxes.cls.cpu().numpy().astype(np.int32)
    
    class_to_color = {}
    max_hues = 180
    hues = np.linspace(0, max_hues, 8, endpoint=False) 
    
    for cls in range(8): 
        hsv_color = np.uint8([[[hues[cls], 200, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        class_to_color[cls] = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))

    for mask, cls_id, box in zip(masks, classes, boxes):
        points = np.array(mask, dtype=np.int32)
        color = class_to_color.get(cls_id, (255, 255, 255))

        cv2.fillPoly(overlay, [points], color)
        cv2.polylines(overlay, [points], True, (255, 255, 255), 1)

        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        M = cv2.moments(points)
        if M["m00"] != 0:
            cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        else:
            cX, cY = int(np.mean(points[:, 0])), int(np.mean(points[:, 1]))

        label = str(cls_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        (tw, th), bl = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(overlay, (cX-tw//2-2, cY-th//2-2), (cX+tw//2+2, cY+th//2+bl+2), (0,0,0), -1)
        cv2.putText(overlay, label, (cX-tw//2, cY+th//2), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)

    final_img = cv2.addWeighted(overlay, alpha, orig_img, 1 - alpha, 0)

    plt.figure(figsize=(14, 12))
    plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def segment_teeth(image_path, conf=0.5):
    model = YOLO("../models/teeth segmentation/best.pt")

    results = model.predict(
        source=image_path,  
        conf=conf,
        verbose=False
    )

    result = results[0]

    return result