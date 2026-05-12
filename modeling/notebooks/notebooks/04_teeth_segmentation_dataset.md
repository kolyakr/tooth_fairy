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
TRAIN_IMAGES_DIST = Path("../data/teeth_segmentation/train/images")
TRAIN_LABELS_DIST = Path("../data/teeth_segmentation/train/labels")
```


```python
TRAIN_IMAGES_SRC = Path("../data/_raw/segmentation/Teeth Segmentation JSON/d2/img")
TRAIN_LABELS_SRC = Path("../data/_raw/segmentation/Teeth Segmentation JSON/d2/ann")

all_images = list(TRAIN_IMAGES_SRC.glob("*.jpg"))
all_labels = list(TRAIN_LABELS_SRC.glob("*.json"))

all_images.sort(key=lambda x: int(x.name.split(".")[0]))
all_labels.sort(key=lambda x: int(x.name.split(".")[0]))

all_images[:2], all_labels[:2]
```




    ([PosixPath('../data/_raw/segmentation/Teeth Segmentation JSON/d2/img/1.jpg'),
      PosixPath('../data/_raw/segmentation/Teeth Segmentation JSON/d2/img/3.jpg')],
     [PosixPath('../data/_raw/segmentation/Teeth Segmentation JSON/d2/ann/1.jpg.json'),
      PosixPath('../data/_raw/segmentation/Teeth Segmentation JSON/d2/ann/3.jpg.json')])




```python
labels = []

for label in all_labels:
    with open(label, "r", encoding="utf8") as f:
        label = json.load(f)

    teeth = {}

    for tooth in label["objects"]:
        teeth[tooth["classTitle"]] = tooth["points"]["exterior"]
    
    labels.append(teeth)

all_labels = labels
```


```python
len(all_labels), len(all_images)
```




    (597, 597)




```python
def plot_full_diagnostic(image_path, all_labels_dict, padding=40):
    result = segment_quadrants(image_path, conf=0.01)
    
    processed_image = result.orig_img.copy()
    h_img, w_img = processed_image.shape[:2]
    filename = getattr(image_path, 'name', str(image_path))
    
    if result.masks is None:
        print(f"No quadrants detected in {filename}")
        return

    masks = result.masks.xy
    boxes = result.boxes.xyxy.cpu().numpy().astype(np.int32)
    classes = result.boxes.cls.cpu().numpy().astype(np.int32)

    quad_colors = [(0, 255, 0), (0, 165, 255), (255, 150, 0), (255, 0, 255)]
    tooth_yellow = (0, 255, 255)
    alpha = 0.35
    
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
    for mask, cls_id in zip(masks, classes):
        pts = np.array(mask, dtype=np.int32)
        color = quad_colors[int(cls_id)] if int(cls_id) < 4 else (255, 255, 255)
        cv2.fillPoly(overlay_pano, [pts], color)

    for t_id, t_points in all_labels_dict.items():
        pts = np.array(t_points, dtype=np.int32)
        cv2.fillPoly(overlay_pano, [pts], tooth_yellow)
        cv2.polylines(overlay_pano, [pts], True, (255, 255, 255), 1)

    combined_pano = cv2.addWeighted(overlay_pano, alpha, processed_image, 1 - alpha, 0)

    for box, cls_id in zip(boxes, classes):
        x1, y1, x2, y2 = box
        color = quad_colors[int(cls_id)] if int(cls_id) < 4 else (255, 255, 255)
        cv2.rectangle(combined_pano, (x1, y1), (x2, y2), color, 3)
        cv2.putText(combined_pano, f"Q{cls_id}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(cv2.cvtColor(combined_pano, cv2.COLOR_BGR2RGB))
    ax0.set_title(f"Full Analysis: {filename}", fontsize=20, fontweight='bold')
    ax0.axis('off')

    grid_pos = {0: (1, 0), 1: (1, 1), 2: (2, 0), 3: (2, 1)}

    for box, cls_id in zip(boxes, classes):
        if int(cls_id) not in grid_pos: continue 
        
        x1, y1, x2, y2 = box
        
        # --- NEW PADDING LOGIC ---
        # Apply padding and ensure we stay within [0, img_size]
        px1 = max(0, x1 - padding)
        py1 = max(0, y1 - padding)
        px2 = min(w_img, x2 + padding)
        py2 = min(h_img, y2 + padding)
        
        crop = processed_image[py1:py2, px1:px2].copy()
        # -------------------------

        quad_color = quad_colors[int(cls_id)]
        crop_overlay = crop.copy()
        target_teeth = quad_map.get(str(cls_id), [])
        found_count = 0
        valid_tooth_points = []

        for t_id in target_teeth:
            if t_id in all_labels_dict:
                found_count += 1
                pts = np.array(all_labels_dict[t_id], dtype=np.int32)
                pts_local = pts.copy()
                
                pts_local[:, 0] -= px1
                pts_local[:, 1] -= py1
                
                cv2.fillPoly(crop_overlay, [pts_local], quad_color)
                valid_tooth_points.append((t_id, pts_local))

        crop = cv2.addWeighted(crop_overlay, alpha, crop, 1 - alpha, 0)

        for t_id, pts_local in valid_tooth_points:
            cv2.polylines(crop, [pts_local], True, (255, 255, 255), 1)
            cX, cY = int(pts_local[:, 0].mean()), int(pts_local[:, 1].mean())
            cv2.putText(crop, str(t_id), (cX - 12, cY + 7), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        row, col = grid_pos[int(cls_id)]
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Quadrant {cls_id} | Teeth Found: {found_count}", fontsize=14)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
```


```python
for label, image in zip(all_labels[:3], all_images[:3]):
    plot_full_diagnostic(image, label, padding=50)
```


    
![png](notebooks/04_teeth_segmentation_dataset_files/notebooks/04_teeth_segmentation_dataset_6_0.png)
    



    
![png](notebooks/04_teeth_segmentation_dataset_files/notebooks/04_teeth_segmentation_dataset_6_1.png)
    



    
![png](notebooks/04_teeth_segmentation_dataset_files/notebooks/04_teeth_segmentation_dataset_6_2.png)
    



```python
def prepare_stage2_data(image_path, all_labels_dict, padding=50):
    IMAGES_DIR = Path("../data/teeth_segmentation/train/images")
    LABELS_DIR = Path("../data/teeth_segmentation/train/labels")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    base_name = Path(image_path).stem
    
    result = segment_quadrants(image_path, conf=0.01)
    if result.masks is None:
        return

    processed_image = result.orig_img 
    h_img, w_img = processed_image.shape[:2]

    boxes = result.boxes.xyxy.cpu().numpy().astype(np.int32)
    classes = result.boxes.cls.cpu().numpy().astype(np.int32)

    available_keys = [str(i) for i in range(1, 33)]
    quad_map = {
        "0": [k for k in available_keys if 1 <= int(k) <= 8],   
        "1": [k for k in available_keys if 9 <= int(k) <= 16],  
        "2": [k for k in available_keys if 17 <= int(k) <= 24], 
        "3": [k for k in available_keys if 25 <= int(k) <= 32], 
    }

    for box, cls_id in zip(boxes, classes):
        q_idx = int(cls_id)
        
        # Dynamic indexing to prevent overwriting
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

        if img_filename.exists():
            continue

        x1, y1, x2, y2 = box
        px1, py1 = max(0, x1 - padding), max(0, y1 - padding)
        px2, py2 = min(w_img, x2 + padding), min(h_img, y2 + padding)
        
        crop = processed_image[py1:py2, px1:px2]
        crop_h, crop_w = crop.shape[:2]

        yolo_labels = []
        target_teeth = quad_map.get(str(q_idx), [])

        for t_id in target_teeth:
            if t_id in all_labels_dict:
                t_num = int(t_id)
                
                # --- MIRRORED 8-CLASS MAPPING (Midline to Molar) ---
                if q_idx == 0:   # Upper Right (1-8): 8 is midline, 1 is molar
                    class_id = 8 - t_num
                elif q_idx == 1: # Upper Left (9-16): 9 is midline, 16 is molar
                    class_id = t_num - 9
                elif q_idx == 2: # Lower Left (17-24): 24 is midline, 17 is molar
                    class_id = 24 - t_num
                elif q_idx == 3: # Lower Right (25-32): 25 is midline, 32 is molar
                    class_id = t_num - 25
                
                # Double check bounds (should be 0-7)
                class_id = int(np.clip(class_id, 0, 7))
                
                pts = np.array(all_labels_dict[t_id], dtype=np.float32)
                pts[:, 0] -= px1
                pts[:, 1] -= py1
                pts[:, 0] /= crop_w
                pts[:, 1] /= crop_h
                pts = np.clip(pts, 0, 1)

                line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in pts.flatten()])
                yolo_labels.append(line)

        if yolo_labels:
            cv2.imwrite(str(img_filename), crop)
            with open(txt_filename, "w") as f:
                f.write("\n".join(yolo_labels))

    print(f"Processed: {base_name}")
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

    Processed: 1
    Processed: 3
    Processed: 4
    Processed: 5
    Processed: 6
    Processed: 7
    Processed: 8
    Processed: 9
    Processed: 10
    Processed: 11
    Processed: 12
    Processed: 13
    Processed: 14
    Processed: 15
    Processed: 16
    Processed: 17
    Processed: 18
    Processed: 19
    Processed: 20
    Processed: 21
    Processed: 22
    Processed: 23
    Processed: 24
    Processed: 25
    Processed: 26
    Processed: 27
    Processed: 28
    Processed: 29
    Processed: 30
    Processed: 31
    Processed: 32
    Processed: 33
    Processed: 34
    Processed: 35
    Processed: 36
    Processed: 37
    Processed: 38
    Processed: 39
    Processed: 40
    Processed: 41
    Processed: 42
    Processed: 43
    Processed: 44
    Processed: 45
    Processed: 46
    Processed: 47
    Processed: 48
    Processed: 49
    Processed: 50
    Processed: 51
    Processed: 52
    Processed: 53
    Processed: 54
    Processed: 55
    Processed: 56
    Processed: 57
    Processed: 58
    Processed: 59
    Processed: 60
    Processed: 61
    Processed: 62
    Processed: 63
    Processed: 64
    Processed: 65
    Processed: 66
    Processed: 67
    Processed: 68
    Processed: 69
    Processed: 70
    Processed: 71
    Processed: 72
    Processed: 73
    Processed: 74
    Processed: 75
    Processed: 76
    Processed: 77
    Processed: 78
    Processed: 79
    Processed: 80
    Processed: 81
    Processed: 82
    Processed: 83
    Processed: 84
    Processed: 85
    Processed: 86
    Processed: 87
    Processed: 88
    Processed: 89
    Processed: 90
    Processed: 91
    Processed: 92
    Processed: 93
    Processed: 94
    Processed: 95
    Processed: 96
    Processed: 97
    Processed: 98
    Processed: 99
    Processed: 100
    Processed: 101
    Processed: 102
    Processed: 103
    Processed: 104
    Processed: 105
    Processed: 106
    Processed: 107
    Processed: 108
    Processed: 109
    Processed: 110
    Processed: 111
    Processed: 112
    Processed: 113
    Processed: 114
    Processed: 115
    Processed: 116
    Processed: 117
    Processed: 118
    Processed: 119
    Processed: 120
    Processed: 121
    Processed: 122
    Processed: 123
    Processed: 124
    Processed: 125
    Processed: 126
    Processed: 127
    Processed: 128
    Processed: 129
    Processed: 130
    Processed: 131
    Processed: 132
    Processed: 133
    Processed: 134
    Skipping filename: 135
    Processed: 136
    Processed: 137
    Processed: 138
    Processed: 139
    Processed: 140
    Processed: 141
    Processed: 142
    Processed: 143
    Processed: 144
    Processed: 145
    Processed: 146
    Processed: 147
    Processed: 148
    Processed: 149
    Processed: 150
    Processed: 151
    Processed: 152
    Processed: 153
    Processed: 154
    Processed: 155
    Processed: 156
    Processed: 157
    Processed: 158
    Processed: 159
    Processed: 160
    Processed: 161
    Processed: 162
    Processed: 163
    Processed: 164
    Processed: 165
    Processed: 166
    Processed: 167
    Processed: 168
    Processed: 169
    Processed: 170
    Processed: 171
    Processed: 172
    Processed: 173
    Processed: 174
    Processed: 175
    Processed: 176
    Processed: 177
    Processed: 178
    Processed: 179
    Processed: 180
    Processed: 181
    Processed: 182
    Processed: 183
    Processed: 184
    Processed: 185
    Processed: 186
    Processed: 187
    Processed: 188
    Processed: 189
    Processed: 190
    Processed: 191
    Processed: 192
    Processed: 193
    Processed: 194
    Processed: 195
    Processed: 196
    Processed: 197
    Processed: 198
    Processed: 199
    Processed: 200
    Processed: 201
    Skipping filename: 202
    Processed: 203
    Processed: 204
    Processed: 205
    Processed: 206
    Processed: 207
    Processed: 208
    Processed: 209
    Processed: 210
    Processed: 211
    Skipping filename: 212
    Processed: 213
    Skipping filename: 214
    Skipping filename: 215
    Processed: 216
    Processed: 217
    Processed: 218
    Processed: 219
    Processed: 220
    Processed: 221
    Processed: 222
    Processed: 223
    Processed: 224
    Processed: 225
    Processed: 226
    Processed: 227
    Processed: 228
    Processed: 229
    Processed: 230
    Processed: 231
    Processed: 232
    Processed: 233
    Processed: 234
    Processed: 235
    Processed: 236
    Processed: 237
    Processed: 238
    Processed: 239
    Processed: 240
    Processed: 241
    Processed: 242
    Processed: 243
    Processed: 244
    Processed: 245
    Processed: 246
    Processed: 247
    Processed: 248
    Processed: 249
    Processed: 250
    Processed: 251
    Processed: 252
    Processed: 253
    Processed: 254
    Processed: 255
    Processed: 256
    Processed: 257
    Processed: 258
    Processed: 259
    Processed: 260
    Processed: 261
    Processed: 262
    Processed: 263
    Processed: 264
    Processed: 265
    Processed: 266
    Processed: 267
    Processed: 268
    Processed: 269
    Processed: 270
    Skipping filename: 271
    Processed: 272
    Processed: 273
    Processed: 274
    Processed: 275
    Processed: 276
    Skipping filename: 277
    Processed: 278
    Processed: 279
    Processed: 280
    Processed: 281
    Processed: 282
    Processed: 283
    Processed: 284
    Processed: 285
    Processed: 286
    Processed: 287
    Processed: 288
    Processed: 289
    Processed: 290
    Processed: 291
    Processed: 292
    Skipping filename: 293
    Processed: 294
    Processed: 295
    Processed: 296
    Processed: 297
    Processed: 298
    Processed: 299
    Processed: 300
    Processed: 301
    Processed: 302
    Processed: 303
    Processed: 304
    Processed: 305
    Processed: 306
    Processed: 307
    Processed: 308
    Processed: 309
    Processed: 310
    Skipping filename: 311
    Processed: 312
    Processed: 313
    Processed: 314
    Processed: 315
    Processed: 316
    Processed: 317
    Processed: 318
    Processed: 319
    Processed: 320
    Processed: 321
    Processed: 322
    Processed: 323
    Processed: 324
    Processed: 325
    Processed: 326
    Processed: 327
    Processed: 328
    Processed: 329
    Processed: 330
    Processed: 331
    Processed: 332
    Processed: 333
    Processed: 334
    Processed: 335
    Processed: 336
    Processed: 337
    Processed: 338
    Processed: 339
    Processed: 340
    Processed: 341
    Processed: 342
    Processed: 343
    Processed: 344
    Processed: 345
    Processed: 346
    Processed: 347
    Processed: 348
    Processed: 349
    Processed: 350
    Processed: 351
    Processed: 352
    Processed: 353
    Processed: 354
    Processed: 355
    Processed: 356
    Processed: 357
    Processed: 358
    Processed: 359
    Processed: 360
    Processed: 361
    Processed: 362
    Processed: 363
    Processed: 364
    Processed: 365
    Processed: 366
    Skipping filename: 367
    Processed: 368
    Processed: 369
    Processed: 370
    Processed: 371
    Processed: 372
    Processed: 373
    Processed: 374
    Processed: 375
    Processed: 376
    Processed: 377
    Processed: 378
    Processed: 379
    Processed: 380
    Processed: 381
    Processed: 382
    Processed: 383
    Processed: 384
    Processed: 385
    Processed: 386
    Processed: 387
    Processed: 388
    Processed: 389
    Processed: 390
    Processed: 391
    Processed: 392
    Skipping filename: 393
    Processed: 394
    Processed: 395
    Processed: 396
    Processed: 397
    Processed: 398
    Processed: 399
    Processed: 400
    Processed: 401
    Skipping filename: 402
    Processed: 403
    Processed: 404
    Processed: 405
    Processed: 406
    Processed: 407
    Processed: 408
    Processed: 409
    Processed: 410
    Processed: 411
    Processed: 412
    Processed: 413
    Processed: 414
    Processed: 415
    Processed: 416
    Processed: 417
    Processed: 418
    Processed: 419
    Processed: 420
    Processed: 421
    Processed: 422
    Skipping filename: 423
    Processed: 424
    Processed: 425
    Processed: 426
    Processed: 427
    Processed: 428
    Processed: 429
    Processed: 430
    Processed: 431
    Processed: 432
    Processed: 433
    Processed: 434
    Processed: 435
    Processed: 436
    Processed: 437
    Processed: 438
    Processed: 439
    Processed: 440
    Processed: 441
    Processed: 442
    Processed: 443
    Processed: 444
    Processed: 445
    Processed: 446
    Processed: 447
    Processed: 448
    Processed: 449
    Processed: 450
    Processed: 451
    Skipping filename: 452
    Processed: 453
    Processed: 454
    Processed: 455
    Processed: 456
    Processed: 457
    Processed: 458
    Processed: 459
    Processed: 460
    Processed: 461
    Processed: 462
    Processed: 463
    Processed: 464
    Processed: 465
    Processed: 466
    Processed: 467
    Processed: 468
    Processed: 469
    Processed: 470
    Processed: 471
    Processed: 472
    Processed: 473
    Processed: 474
    Processed: 475
    Processed: 476
    Processed: 477
    Processed: 478
    Processed: 479
    Processed: 480
    Processed: 481
    Processed: 482
    Processed: 483
    Processed: 484
    Processed: 485
    Processed: 486
    Processed: 487
    Processed: 488
    Processed: 489
    Processed: 490
    Processed: 491
    Processed: 492
    Processed: 493
    Processed: 494
    Skipping filename: 495
    Processed: 496
    Processed: 497
    Processed: 498
    Processed: 499
    Processed: 500
    Processed: 501
    Processed: 502
    Processed: 503
    Processed: 504
    Processed: 505
    Processed: 506
    Processed: 507
    Processed: 508
    Processed: 509
    Processed: 510
    Processed: 511
    Processed: 512
    Processed: 513
    Processed: 514
    Processed: 515
    Processed: 516
    Processed: 517
    Processed: 518
    Processed: 519
    Processed: 520
    Processed: 521
    Processed: 522
    Processed: 523
    Processed: 524
    Processed: 525
    Processed: 526
    Processed: 527
    Processed: 528
    Processed: 529
    Processed: 530
    Processed: 531
    Processed: 532
    Skipping filename: 533
    Processed: 534
    Processed: 535
    Processed: 536
    Processed: 537
    Processed: 538
    Processed: 539
    Processed: 540
    Processed: 541
    Processed: 542
    Skipping filename: 543
    Processed: 544
    Processed: 545
    Processed: 546
    Processed: 547
    Processed: 548
    Processed: 549
    Processed: 550
    Processed: 551
    Processed: 552
    Processed: 553
    Processed: 554
    Processed: 555
    Processed: 556
    Processed: 557
    Processed: 558
    Processed: 559
    Processed: 560
    Processed: 561
    Processed: 562
    Processed: 563
    Processed: 564
    Processed: 565
    Processed: 566
    Processed: 567
    Processed: 568
    Processed: 569
    Processed: 570
    Processed: 571
    Processed: 572
    Processed: 573
    Processed: 574
    Processed: 575
    Processed: 576
    Processed: 577
    Processed: 578
    Processed: 579
    Processed: 580
    Processed: 581
    Processed: 582
    Processed: 583
    Processed: 584
    Processed: 585
    Processed: 586
    Processed: 587
    Processed: 588
    Processed: 589
    Processed: 590
    Processed: 591
    Processed: 592
    Processed: 593
    Processed: 594
    Skipping filename: 595
    Processed: 596
    Processed: 597
    Processed: 598



```python
TRAIN_IMAGES_SRC = Path("../data/_raw/segmentation/dentex_data/training_data/quadrant_enumeration/xrays")
TRAIN_LABELS_SRC = Path("../data/_raw/segmentation/dentex_data/training_data/quadrant_enumeration/train_quadrant_enumeration.json")

all_images = list(TRAIN_IMAGES_SRC.glob("*.png"))

all_images.sort(key=lambda x: int(x.name.split(".")[0].split("_")[1]))

all_images[:10]
```




    [PosixPath('../data/_raw/segmentation/dentex_data/training_data/quadrant_enumeration/xrays/train_0.png'),
     PosixPath('../data/_raw/segmentation/dentex_data/training_data/quadrant_enumeration/xrays/train_1.png'),
     PosixPath('../data/_raw/segmentation/dentex_data/training_data/quadrant_enumeration/xrays/train_2.png'),
     PosixPath('../data/_raw/segmentation/dentex_data/training_data/quadrant_enumeration/xrays/train_3.png'),
     PosixPath('../data/_raw/segmentation/dentex_data/training_data/quadrant_enumeration/xrays/train_4.png'),
     PosixPath('../data/_raw/segmentation/dentex_data/training_data/quadrant_enumeration/xrays/train_5.png'),
     PosixPath('../data/_raw/segmentation/dentex_data/training_data/quadrant_enumeration/xrays/train_6.png'),
     PosixPath('../data/_raw/segmentation/dentex_data/training_data/quadrant_enumeration/xrays/train_7.png'),
     PosixPath('../data/_raw/segmentation/dentex_data/training_data/quadrant_enumeration/xrays/train_8.png'),
     PosixPath('../data/_raw/segmentation/dentex_data/training_data/quadrant_enumeration/xrays/train_9.png')]




```python
with open(TRAIN_LABELS_SRC, "r", encoding="utf8") as f:
    data = json.load(f)

data.keys()
```




    dict_keys(['images', 'annotations', 'categories_1', 'categories_2'])




```python
data["categories_1"], data["categories_2"]
```




    ([{'id': 0, 'name': 1, 'supercategory': 1},
      {'id': 1, 'name': 2, 'supercategory': 2},
      {'id': 2, 'name': 3, 'supercategory': 3},
      {'id': 3, 'name': 4, 'supercategory': 4}],
     [{'id': 0, 'name': '1', 'supercategory': '1'},
      {'id': 1, 'name': '2', 'supercategory': '2'},
      {'id': 2, 'name': '3', 'supercategory': '3'},
      {'id': 3, 'name': '4', 'supercategory': '4'},
      {'id': 4, 'name': '5', 'supercategory': '5'},
      {'id': 5, 'name': '6', 'supercategory': '6'},
      {'id': 6, 'name': '7', 'supercategory': '7'},
      {'id': 7, 'name': '8', 'supercategory': '8'}])




```python
data["annotations"][0]
```




    {'iscrowd': 0,
     'image_id': 1,
     'bbox': [1283.3333333333333, 459.25925925925924, 100.0, 262.03703703703707],
     'segmentation': [[1366,
       459,
       1383,
       662,
       1380,
       716,
       1295,
       721,
       1288,
       659,
       1283,
       464]],
     'id': 1,
     'area': 22904,
     'category_id_1': 0,
     'category_id_2': 0}




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
    idx = filename_to_idx[fname]
    
    cls = convert_to_global_class(teeth["category_id_1"], teeth["category_id_2"])
    seg = np.array(teeth["segmentation"], dtype=np.int32).reshape(-1, 2)
    all_labels[idx][str(cls)] = seg
```


```python
len(all_labels), len(all_images)
```




    (634, 634)




```python
for label, image in zip(all_labels[:3], all_images[:3]):
    plot_full_diagnostic(image, label, padding=50)
```


    
![png](notebooks/04_teeth_segmentation_dataset_files/notebooks/04_teeth_segmentation_dataset_18_0.png)
    



    
![png](notebooks/04_teeth_segmentation_dataset_files/notebooks/04_teeth_segmentation_dataset_18_1.png)
    



    
![png](notebooks/04_teeth_segmentation_dataset_files/notebooks/04_teeth_segmentation_dataset_18_2.png)
    



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

    Processed: train_0
    Processed: train_1
    Processed: train_2
    Processed: train_3
    Processed: train_4
    Processed: train_5
    Processed: train_6
    Processed: train_7
    Processed: train_8
    Processed: train_9
    Processed: train_10
    Processed: train_11
    Processed: train_12
    Processed: train_13
    Processed: train_14
    Processed: train_15
    Processed: train_16
    Processed: train_17
    Processed: train_18
    Processed: train_19
    Processed: train_20
    Processed: train_21
    Processed: train_22
    Processed: train_23
    Skipping filename: 24
    Processed: train_25
    Processed: train_26
    Processed: train_27
    Processed: train_28
    Processed: train_29
    Processed: train_30
    Processed: train_31
    Processed: train_32
    Processed: train_33
    Processed: train_34
    Processed: train_35
    Processed: train_36
    Processed: train_37
    Processed: train_38
    Processed: train_39
    Processed: train_40
    Processed: train_41
    Processed: train_42
    Processed: train_43
    Processed: train_44
    Processed: train_45
    Processed: train_46
    Processed: train_47
    Processed: train_48
    Processed: train_49
    Processed: train_50
    Processed: train_51
    Processed: train_52
    Processed: train_53
    Processed: train_54
    Processed: train_55
    Processed: train_56
    Processed: train_57
    Processed: train_58
    Processed: train_59
    Processed: train_60
    Processed: train_61
    Processed: train_62
    Processed: train_63
    Processed: train_64
    Processed: train_65
    Processed: train_66
    Processed: train_67
    Processed: train_68
    Processed: train_69
    Skipping filename: 70
    Processed: train_71
    Processed: train_72
    Processed: train_73
    Processed: train_74
    Processed: train_75
    Processed: train_76
    Processed: train_77
    Processed: train_78
    Processed: train_79
    Processed: train_80
    Processed: train_81
    Processed: train_82
    Processed: train_83
    Processed: train_84
    Processed: train_85
    Processed: train_86
    Processed: train_87
    Processed: train_88
    Processed: train_89
    Processed: train_90
    Processed: train_91
    Processed: train_92
    Processed: train_93
    Processed: train_94
    Processed: train_95
    Processed: train_96
    Processed: train_97
    Processed: train_98
    Processed: train_99
    Processed: train_100
    Processed: train_101
    Processed: train_102
    Processed: train_103
    Processed: train_104
    Processed: train_105
    Processed: train_106
    Processed: train_107
    Processed: train_108
    Processed: train_109
    Processed: train_110
    Processed: train_111
    Processed: train_112
    Processed: train_113
    Processed: train_114
    Processed: train_115
    Processed: train_116
    Processed: train_117
    Processed: train_118
    Processed: train_119
    Processed: train_120
    Processed: train_121
    Processed: train_122
    Processed: train_123
    Processed: train_124
    Processed: train_125
    Processed: train_126
    Processed: train_127
    Processed: train_128
    Processed: train_129
    Processed: train_130
    Processed: train_131
    Processed: train_132
    Skipping filename: 133
    Processed: train_134
    Processed: train_135
    Processed: train_136
    Processed: train_137
    Processed: train_138
    Processed: train_139
    Processed: train_140
    Processed: train_141
    Processed: train_142
    Processed: train_143
    Processed: train_144
    Processed: train_145
    Processed: train_146
    Processed: train_147
    Processed: train_148
    Processed: train_149
    Skipping filename: 150
    Processed: train_151
    Processed: train_152
    Processed: train_153
    Processed: train_154
    Processed: train_155
    Processed: train_156
    Processed: train_157
    Processed: train_158
    Processed: train_159
    Processed: train_160
    Processed: train_161
    Processed: train_162
    Processed: train_163
    Processed: train_164
    Processed: train_165
    Processed: train_166
    Processed: train_167
    Processed: train_168
    Processed: train_169
    Processed: train_170
    Processed: train_171
    Processed: train_172
    Processed: train_173
    Processed: train_174
    Processed: train_175
    Processed: train_176
    Processed: train_177
    Processed: train_178
    Processed: train_179
    Processed: train_180
    Processed: train_181
    Processed: train_182
    Processed: train_183
    Processed: train_184
    Processed: train_185
    Processed: train_186
    Processed: train_187
    Processed: train_188
    Processed: train_189
    Processed: train_190
    Processed: train_191
    Processed: train_192
    Processed: train_193
    Processed: train_194
    Processed: train_195
    Processed: train_196
    Processed: train_197
    Processed: train_198
    Processed: train_199
    Processed: train_200
    Processed: train_201
    Processed: train_202
    Processed: train_203
    Processed: train_204
    Processed: train_205
    Processed: train_206
    Processed: train_207
    Processed: train_208
    Processed: train_209
    Processed: train_210
    Processed: train_211
    Processed: train_212
    Processed: train_213
    Processed: train_214
    Processed: train_215
    Processed: train_216
    Processed: train_217
    Processed: train_218
    Processed: train_219
    Processed: train_220
    Processed: train_221
    Processed: train_222
    Processed: train_223
    Processed: train_224
    Processed: train_225
    Processed: train_226
    Processed: train_227
    Processed: train_228
    Processed: train_229
    Processed: train_230
    Processed: train_231
    Processed: train_232
    Processed: train_233
    Processed: train_234
    Processed: train_235
    Processed: train_236
    Processed: train_237
    Processed: train_238
    Processed: train_239
    Processed: train_240
    Processed: train_241
    Processed: train_242
    Processed: train_243
    Processed: train_244
    Processed: train_245
    Processed: train_246
    Processed: train_247
    Processed: train_248
    Processed: train_249
    Processed: train_250
    Processed: train_251
    Processed: train_252
    Processed: train_253
    Processed: train_254
    Processed: train_255
    Processed: train_256
    Processed: train_257
    Processed: train_258
    Processed: train_259
    Processed: train_260
    Processed: train_261
    Processed: train_262
    Processed: train_263
    Processed: train_264
    Skipping filename: 265
    Processed: train_266
    Processed: train_267
    Processed: train_268
    Processed: train_269
    Processed: train_270
    Processed: train_271
    Processed: train_272
    Processed: train_273
    Processed: train_274
    Processed: train_275
    Processed: train_276
    Processed: train_277
    Processed: train_278
    Processed: train_279
    Processed: train_280
    Processed: train_281
    Processed: train_282
    Processed: train_283
    Processed: train_284
    Processed: train_285
    Processed: train_286
    Processed: train_287
    Processed: train_288
    Processed: train_289
    Processed: train_290
    Processed: train_291
    Processed: train_292
    Processed: train_293
    Processed: train_294
    Processed: train_295
    Processed: train_296
    Processed: train_297
    Processed: train_298
    Processed: train_299
    Processed: train_300
    Processed: train_301
    Processed: train_302
    Processed: train_303
    Processed: train_304
    Processed: train_305
    Processed: train_306
    Processed: train_307
    Processed: train_308
    Processed: train_309
    Processed: train_310
    Processed: train_311
    Processed: train_312
    Processed: train_313
    Processed: train_314
    Processed: train_315
    Processed: train_316
    Processed: train_317
    Processed: train_318
    Processed: train_319
    Processed: train_320
    Processed: train_321
    Processed: train_322
    Processed: train_323
    Processed: train_324
    Processed: train_325
    Processed: train_326
    Processed: train_327
    Processed: train_328
    Processed: train_329
    Processed: train_330
    Processed: train_331
    Processed: train_332
    Processed: train_333
    Processed: train_334
    Processed: train_335
    Processed: train_336
    Processed: train_337
    Processed: train_338
    Processed: train_339
    Processed: train_340
    Processed: train_341
    Processed: train_342
    Processed: train_343
    Processed: train_344
    Processed: train_345
    Processed: train_346
    Processed: train_347
    Processed: train_348
    Processed: train_349
    Processed: train_350
    Processed: train_351
    Processed: train_352
    Processed: train_353
    Processed: train_354
    Processed: train_355
    Processed: train_356
    Processed: train_357
    Processed: train_358
    Processed: train_359
    Processed: train_360
    Processed: train_361
    Processed: train_362
    Processed: train_363
    Processed: train_364
    Processed: train_365
    Processed: train_366
    Processed: train_367
    Processed: train_368
    Processed: train_369
    Processed: train_370
    Processed: train_371
    Processed: train_372
    Processed: train_373
    Processed: train_374
    Processed: train_375
    Processed: train_376
    Processed: train_377
    Processed: train_378
    Processed: train_379
    Processed: train_380
    Processed: train_381
    Processed: train_382
    Processed: train_383
    Processed: train_384
    Processed: train_385
    Processed: train_386
    Processed: train_387
    Processed: train_388
    Processed: train_389
    Processed: train_390
    Processed: train_391
    Processed: train_392
    Processed: train_393
    Processed: train_394
    Processed: train_395
    Processed: train_396
    Processed: train_397
    Processed: train_398
    Processed: train_399
    Processed: train_400
    Processed: train_401
    Processed: train_402
    Processed: train_403
    Processed: train_404
    Processed: train_405
    Processed: train_406
    Processed: train_407
    Processed: train_408
    Processed: train_409
    Processed: train_410
    Processed: train_411
    Processed: train_412
    Processed: train_413
    Processed: train_414
    Processed: train_415
    Processed: train_416
    Processed: train_417
    Processed: train_418
    Processed: train_419
    Processed: train_420
    Processed: train_421
    Processed: train_422
    Processed: train_423
    Processed: train_424
    Processed: train_425
    Processed: train_426
    Processed: train_427
    Processed: train_428
    Processed: train_429
    Processed: train_430
    Processed: train_431
    Processed: train_432
    Processed: train_433
    Processed: train_434
    Processed: train_435
    Processed: train_436
    Processed: train_437
    Processed: train_438
    Processed: train_439
    Processed: train_440
    Processed: train_441
    Processed: train_442
    Processed: train_443
    Processed: train_444
    Processed: train_445
    Processed: train_446
    Processed: train_447
    Processed: train_448
    Processed: train_449
    Processed: train_450
    Processed: train_451
    Processed: train_452
    Processed: train_453
    Processed: train_454
    Processed: train_455
    Processed: train_456
    Processed: train_457
    Processed: train_458
    Processed: train_459
    Processed: train_460
    Processed: train_461
    Processed: train_462
    Processed: train_463
    Processed: train_464
    Processed: train_465
    Processed: train_466
    Processed: train_467
    Processed: train_468
    Processed: train_469
    Processed: train_470
    Processed: train_471
    Processed: train_472
    Processed: train_473
    Processed: train_474
    Processed: train_475
    Processed: train_476
    Processed: train_477
    Processed: train_478
    Processed: train_479
    Processed: train_480
    Processed: train_481
    Processed: train_482
    Processed: train_483
    Processed: train_484
    Processed: train_485
    Skipping filename: 486
    Processed: train_487
    Processed: train_488
    Processed: train_489
    Processed: train_490
    Processed: train_491
    Processed: train_492
    Processed: train_493
    Processed: train_494
    Processed: train_495
    Processed: train_496
    Processed: train_497
    Processed: train_498
    Processed: train_499
    Processed: train_500
    Processed: train_501
    Processed: train_502
    Processed: train_503
    Processed: train_504
    Processed: train_505
    Processed: train_506
    Skipping filename: 507
    Processed: train_508
    Processed: train_509
    Processed: train_510
    Processed: train_511
    Skipping filename: 512
    Processed: train_513
    Processed: train_514
    Processed: train_515
    Processed: train_516
    Processed: train_517
    Processed: train_518
    Processed: train_519
    Processed: train_520
    Processed: train_521
    Processed: train_522
    Processed: train_523
    Processed: train_524
    Processed: train_525
    Processed: train_526
    Skipping filename: 527
    Processed: train_528
    Processed: train_529
    Processed: train_530
    Processed: train_531
    Processed: train_532
    Processed: train_533
    Processed: train_534
    Processed: train_535
    Processed: train_536
    Processed: train_537
    Processed: train_538
    Processed: train_539
    Processed: train_540
    Processed: train_541
    Processed: train_542
    Processed: train_543
    Processed: train_544
    Processed: train_545
    Processed: train_546
    Processed: train_547
    Processed: train_548
    Processed: train_549
    Processed: train_550
    Processed: train_551
    Processed: train_552
    Processed: train_553
    Processed: train_554
    Processed: train_555
    Processed: train_556
    Processed: train_557
    Processed: train_558
    Processed: train_559
    Processed: train_560
    Processed: train_561
    Processed: train_562
    Processed: train_563
    Processed: train_564
    Processed: train_565
    Processed: train_566
    Processed: train_567
    Processed: train_568
    Processed: train_569
    Processed: train_570
    Processed: train_571
    Processed: train_572
    Processed: train_573
    Processed: train_574
    Processed: train_575
    Processed: train_576
    Processed: train_577
    Processed: train_578
    Processed: train_579
    Processed: train_580
    Processed: train_581
    Processed: train_582
    Processed: train_583
    Skipping filename: 584
    Processed: train_585
    Processed: train_586
    Processed: train_587
    Processed: train_588
    Processed: train_589
    Processed: train_590
    Processed: train_591
    Skipping filename: 592
    Processed: train_593
    Processed: train_594
    Processed: train_595
    Processed: train_596
    Processed: train_597
    Processed: train_598
    Processed: train_599
    Processed: train_600
    Processed: train_601
    Processed: train_602
    Processed: train_603
    Processed: train_604
    Processed: train_605
    Processed: train_606
    Processed: train_607
    Processed: train_608
    Processed: train_609
    Processed: train_610
    Processed: train_611
    Processed: train_612
    Processed: train_613
    Processed: train_614
    Processed: train_615
    Processed: train_616
    Processed: train_617
    Processed: train_618
    Processed: train_619
    Processed: train_620
    Processed: train_621
    Processed: train_622
    Processed: train_623
    Processed: train_624
    Processed: train_625
    Processed: train_626
    Processed: train_627
    Processed: train_628
    Processed: train_629
    Processed: train_630
    Processed: train_631
    Processed: train_632
    Processed: train_633



```python
TRAIN_IMAGES_SRC = Path("../data/_raw/segmentation/Tooth Label in Panoramic X-ray.v2i.coco-segmentation/train")
TRAIN_LABELS_SRC = Path("../data/_raw/segmentation/Tooth Label in Panoramic X-ray.v2i.coco-segmentation/train/_annotations.coco.json")

VAL_IMAGES_SRC = Path("../data/_raw/segmentation/Tooth Label in Panoramic X-ray.v2i.coco-segmentation/valid")
VAL_LABELS_SRC = Path("../data/_raw/segmentation/Tooth Label in Panoramic X-ray.v2i.coco-segmentation/valid/_annotations.coco.json")

TEST_IMAGES_SRC = Path("../data/_raw/segmentation/Tooth Label in Panoramic X-ray.v2i.coco-segmentation/test")
TEST_LABELS_SRC = Path("../data/_raw/segmentation/Tooth Label in Panoramic X-ray.v2i.coco-segmentation/test/_annotations.coco.json")


all_images = list(TRAIN_IMAGES_SRC.glob("*.jpg"))

all_images.sort(key=lambda x: int(x.name.split(".")[0].split("_")[0]))

all_images[:10]
```




    [PosixPath('../data/_raw/segmentation/Tooth Label in Panoramic X-ray.v2i.coco-segmentation/train/4_png.rf.55b10907f9e7bc3f26f47eacbf0fad57.jpg'),
     PosixPath('../data/_raw/segmentation/Tooth Label in Panoramic X-ray.v2i.coco-segmentation/train/6_png.rf.95147ce43edeb589a0a9f5b593d43c5d.jpg'),
     PosixPath('../data/_raw/segmentation/Tooth Label in Panoramic X-ray.v2i.coco-segmentation/train/7_png.rf.1ee1eeb1b62ff1a08bc068cb4d6f5dbb.jpg'),
     PosixPath('../data/_raw/segmentation/Tooth Label in Panoramic X-ray.v2i.coco-segmentation/train/9_png.rf.4048b26b0d52451cdc504f4cea613efc.jpg'),
     PosixPath('../data/_raw/segmentation/Tooth Label in Panoramic X-ray.v2i.coco-segmentation/train/11_png.rf.c3ea9a71e3e94ed4c17aea92c21907c6.jpg'),
     PosixPath('../data/_raw/segmentation/Tooth Label in Panoramic X-ray.v2i.coco-segmentation/train/12_png.rf.3eb9dbb86b21b1904b63bc284063a13c.jpg'),
     PosixPath('../data/_raw/segmentation/Tooth Label in Panoramic X-ray.v2i.coco-segmentation/train/14_png.rf.7d99ad6bfbb074e5cb9eac7e32dda249.jpg'),
     PosixPath('../data/_raw/segmentation/Tooth Label in Panoramic X-ray.v2i.coco-segmentation/train/15_png.rf.6d242cb4329829a4a9fa41582eae26a1.jpg'),
     PosixPath('../data/_raw/segmentation/Tooth Label in Panoramic X-ray.v2i.coco-segmentation/train/19_png.rf.8859e5b061d5edd944d95ab966eae179.jpg'),
     PosixPath('../data/_raw/segmentation/Tooth Label in Panoramic X-ray.v2i.coco-segmentation/train/22_png.rf.93644822b3d078fd139bb7019a4749fd.jpg')]




```python
with open(TRAIN_LABELS_SRC, "r", encoding="utf8") as f:
    data = json.load(f)

data.keys()
```




    dict_keys(['info', 'licenses', 'categories', 'images', 'annotations'])




```python
data["categories"]
```




    [{'id': 0, 'name': 'Teeth-Objects', 'supercategory': 'none'},
     {'id': 1, 'name': '11', 'supercategory': 'Teeth-Objects'},
     {'id': 2, 'name': '12', 'supercategory': 'Teeth-Objects'},
     {'id': 3, 'name': '13', 'supercategory': 'Teeth-Objects'},
     {'id': 4, 'name': '14', 'supercategory': 'Teeth-Objects'},
     {'id': 5, 'name': '15', 'supercategory': 'Teeth-Objects'},
     {'id': 6, 'name': '16', 'supercategory': 'Teeth-Objects'},
     {'id': 7, 'name': '17', 'supercategory': 'Teeth-Objects'},
     {'id': 8, 'name': '18', 'supercategory': 'Teeth-Objects'},
     {'id': 9, 'name': '21', 'supercategory': 'Teeth-Objects'},
     {'id': 10, 'name': '22', 'supercategory': 'Teeth-Objects'},
     {'id': 11, 'name': '23', 'supercategory': 'Teeth-Objects'},
     {'id': 12, 'name': '24', 'supercategory': 'Teeth-Objects'},
     {'id': 13, 'name': '25', 'supercategory': 'Teeth-Objects'},
     {'id': 14, 'name': '26', 'supercategory': 'Teeth-Objects'},
     {'id': 15, 'name': '27', 'supercategory': 'Teeth-Objects'},
     {'id': 16, 'name': '28', 'supercategory': 'Teeth-Objects'},
     {'id': 17, 'name': '31', 'supercategory': 'Teeth-Objects'},
     {'id': 18, 'name': '32', 'supercategory': 'Teeth-Objects'},
     {'id': 19, 'name': '33', 'supercategory': 'Teeth-Objects'},
     {'id': 20, 'name': '34', 'supercategory': 'Teeth-Objects'},
     {'id': 21, 'name': '35', 'supercategory': 'Teeth-Objects'},
     {'id': 22, 'name': '36', 'supercategory': 'Teeth-Objects'},
     {'id': 23, 'name': '37', 'supercategory': 'Teeth-Objects'},
     {'id': 24, 'name': '38', 'supercategory': 'Teeth-Objects'},
     {'id': 25, 'name': '41', 'supercategory': 'Teeth-Objects'},
     {'id': 26, 'name': '42', 'supercategory': 'Teeth-Objects'},
     {'id': 27, 'name': '43', 'supercategory': 'Teeth-Objects'},
     {'id': 28, 'name': '44', 'supercategory': 'Teeth-Objects'},
     {'id': 29, 'name': '45', 'supercategory': 'Teeth-Objects'},
     {'id': 30, 'name': '46', 'supercategory': 'Teeth-Objects'},
     {'id': 31, 'name': '47', 'supercategory': 'Teeth-Objects'},
     {'id': 32, 'name': '48', 'supercategory': 'Teeth-Objects'}]




```python
data["images"][:2]
```




    [{'id': 0,
      'license': 1,
      'file_name': '100_png.rf.02a51c0633e2dadc5d2b5f9f4391d185.jpg',
      'height': 1150,
      'width': 3138,
      'date_captured': '2023-06-01T13:23:19+00:00'},
     {'id': 1,
      'license': 1,
      'file_name': '107_png.rf.056233b71a2f0007f7bd6bf3377b9149.jpg',
      'height': 1300,
      'width': 3126,
      'date_captured': '2023-06-01T13:23:19+00:00'}]




```python
data["annotations"][10]
```




    {'id': 10,
     'image_id': 0,
     'category_id': 12,
     'bbox': [1901, 162, 114, 315],
     'area': 35910,
     'segmentation': [[1963.007,
       162,
       1957.986,
       164.001,
       1954.001,
       168.004,
       1944.995,
       185.001,
       1936.993,
       204.999,
       1927.987,
       240.005,
       1925.006,
       259.003,
       1921.994,
       290.996,
       1914.996,
       326.002,
       1912.988,
       359.996,
       1907.998,
       382.996,
       1904.013,
       392,
       1901,
       403.995,
       1901,
       426.995,
       1903.009,
       436,
       1905.99,
       443.003,
       1911.011,
       449.995,
       1914.996,
       452.996,
       1919.985,
       454.998,
       1945.999,
       472.995,
       1960.999,
       476.997,
       1971.009,
       476.997,
       1985.005,
       472.995,
       2002.985,
       463.002,
       2005.998,
       460,
       2010.015,
       448.995,
       2010.015,
       446.005,
       2004.994,
       436,
       2000.004,
       418.002,
       2001.008,
       402.005,
       2000.004,
       382.996,
       2004.994,
       355.005,
       2005.998,
       326.002,
       2009.01,
       312.996,
       2012.996,
       281.002,
       2014,
       265.995,
       2012.996,
       231,
       2015.004,
       216.005,
       2015.004,
       199.996,
       2011.991,
       188.002,
       2010.015,
       184,
       2005.998,
       179.998,
       1992.002,
       174.995,
       1967.997,
       162]],
     'iscrowd': 0}




```python
len(data["annotations"])
```




    1999




```python
def convert_to_standard_universal(class_id):
    cid = int(class_id)
    
    if 1 <= cid <= 8:
        return 9 - cid
    
    if 17 <= cid <= 24:
        return 41 - cid
        
    return cid
```


```python
labels_src_list = [TRAIN_LABELS_SRC, VAL_LABELS_SRC, TEST_LABELS_SRC]
images_src_list = [TRAIN_IMAGES_SRC, VAL_IMAGES_SRC, TEST_IMAGES_SRC]

total_labels = []
total_images = []

for LABELS_PATH, IMAGES_PATH in zip(labels_src_list, images_src_list):
    with open(LABELS_PATH, "r", encoding="utf8") as f:
        data = json.load(f)
        
    json_images = data["images"]
    id_to_metadata = {img['id']: img for img in json_images}
    
    temp_labels_dict = {img_id: {} for img_id in id_to_metadata.keys()}

    for teeth in data["annotations"]:
        img_id = teeth["image_id"]
        if teeth["category_id"] == 0: continue
            
        cls = convert_to_standard_universal(teeth["category_id"])
        seg = np.array(teeth["segmentation"], dtype=np.int32).flatten().reshape(-1, 2)
        
        if cls:
            temp_labels_dict[img_id][str(cls)] = seg

    current_split_images = []
    current_split_labels = []
    
    for img_id in sorted(id_to_metadata.keys()):
        meta = id_to_metadata[img_id]
        img_path = IMAGES_PATH / meta["file_name"]
        
        if img_path.exists():
            current_split_images.append(img_path)
            current_split_labels.append(temp_labels_dict[img_id])
            
    total_images.extend(current_split_images)
    total_labels.extend(current_split_labels)

all_labels = total_labels
all_images = total_images

len(all_labels), len(all_images)
```




    (113, 113)




```python
for label, image in zip(all_labels[:3], all_images[:3]):
    plot_full_diagnostic(image, label, padding=50)
```


    
![png](notebooks/04_teeth_segmentation_dataset_files/notebooks/04_teeth_segmentation_dataset_29_0.png)
    



    
![png](notebooks/04_teeth_segmentation_dataset_files/notebooks/04_teeth_segmentation_dataset_29_1.png)
    



    
![png](notebooks/04_teeth_segmentation_dataset_files/notebooks/04_teeth_segmentation_dataset_29_2.png)
    



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
    train_dir="../data/teeth_segmentation/train",
    val_dir="../data/teeth_segmentation/val",
    val_ratio=0.15
)
```

    Quadrant   | Total    | Moving to Val
    -----------------------------------
    q2         | 1197     | 179         
    q3         | 1198     | 179         
    q0         | 1185     | 177         
    q1         | 1186     | 177         
    
    Split complete. Validation set is now balanced by quadrant.

