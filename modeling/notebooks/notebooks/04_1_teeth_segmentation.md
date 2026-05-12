```python
import os
import sys

sys.path.append(os.path.abspath("../"))

from pathlib import Path
from utils import visualize_quadrant_crops, get_quadrant_crops, segment_quadrants
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np
import cv2
```


```python
TEST_IMAGES = Path("../data/_raw/classification/Dental OPG XRAY Dataset/Dental OPG (Object Detection)/Original Dataset")
all_images = list(TEST_IMAGES.glob("*.jpg"))
```


```python
orig_img, crops, fname = get_quadrant_crops(all_images[0], padding=40)

visualize_quadrant_crops(orig_img, crops, fname)
```


    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_2_0.png)
    



```python
def visualize_teeth(result, alpha=0.35):
    orig_img = result.orig_img.copy()
    overlay = orig_img.copy()
    
    if result.masks is None:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        axes[0].imshow(img_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        axes[1].imshow(img_rgb)
        axes[1].set_title("No Masks Detected")
        axes[1].axis('off')
        plt.tight_layout()
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

    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    
    axes[0].imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original", fontsize=20, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"{len(classes)} teeth", fontsize=20, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
```


```python
model = YOLO("../models/teeth segmentation/best.pt")

for i in range(len(crops)):
    results = model.predict(
        source=crops[i]["crop"],  
        conf=0.35,
        verbose=False
    )

    result = results[0]

    visualize_teeth(result)
```


    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_4_0.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_4_1.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_4_2.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_4_3.png)
    



```python
for image in all_images:
    orig_img, crops, fname = get_quadrant_crops(image, padding=40)

    visualize_quadrant_crops(orig_img, crops, fname)

    for i in range(len(crops)):
        results = model.predict(
            source=crops[i]["crop"],  
            conf=0.01,
            verbose=False
        )

        result = results[0]

        visualize_teeth(result)

```


    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_0.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_1.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_2.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_3.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_4.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_5.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_6.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_7.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_8.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_9.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_10.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_11.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_12.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_13.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_14.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_15.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_16.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_17.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_18.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_19.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_20.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_21.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_22.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_23.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_24.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_25.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_26.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_27.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_28.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_29.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_30.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_31.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_32.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_33.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_34.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_35.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_36.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_37.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_38.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_39.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_40.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_41.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_42.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_43.png)
    



    
![png](notebooks/04_1_teeth_segmentation_files/notebooks/04_1_teeth_segmentation_5_44.png)
    



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[12], line 2
          1 for image in all_images:
    ----> 2     orig_img, crops, fname = get_quadrant_crops(image, padding=40)
          4     visualize_quadrant_crops(orig_img, crops, fname)
          6     for i in range(len(crops)):


    File ~/My Fucking Stuff/machine learning/code/tooth_fairy/utils/index.py:45, in get_quadrant_crops(image_path, padding)
         44 def get_quadrant_crops(image_path, padding=40):
    ---> 45     result = segment_quadrants(image_path, conf=0.01) 
         46     orig_img = result.orig_img.copy()
         47     h_img, w_img = orig_img.shape[:2]


    File ~/My Fucking Stuff/machine learning/code/tooth_fairy/utils/index.py:14, in segment_quadrants(image_path, conf)
         11 clahe_bgr = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
         13 quadrant_seg_model_path = "../models/quadrant segmentation/best.pt"
    ---> 14 model = YOLO(quadrant_seg_model_path)
         16 results = model.predict(
         17     source=clahe_bgr,  
         18     conf=conf,
         19     verbose=False
         20 )
         22 result = results[0]


    File ~/miniforge3/envs/ml-env/lib/python3.10/site-packages/ultralytics/models/yolo/model.py:76, in YOLO.__init__(self, model, task, verbose)
         73     self.__dict__ = new_instance.__dict__
         74 else:
         75     # Continue with default YOLO initialization
    ---> 76     super().__init__(model=model, task=task, verbose=verbose)
         77     if hasattr(self.model, "model") and "RTDETR" in self.model.model[-1]._get_name():  # if RTDETR head
         78         from ultralytics import RTDETR


    File ~/miniforge3/envs/ml-env/lib/python3.10/site-packages/ultralytics/engine/model.py:144, in Model.__init__(self, model, task, verbose)
        142     self._new(model, task=task, verbose=verbose)
        143 else:
    --> 144     self._load(model, task=task)
        146 # Delete super().training for accessing self.model.training
        147 del self.training


    File ~/miniforge3/envs/ml-env/lib/python3.10/site-packages/ultralytics/engine/model.py:283, in Model._load(self, weights, task)
        280 weights = checks.check_model_file_from_stem(weights)  # add suffix, i.e. yolo26 -> yolo26n.pt
        282 if str(weights).rpartition(".")[-1] == "pt":
    --> 283     self.model, self.ckpt = load_checkpoint(weights)
        284     self.task = self.model.task
        285     self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)


    File ~/miniforge3/envs/ml-env/lib/python3.10/site-packages/ultralytics/nn/tasks.py:1517, in load_checkpoint(weight, device, inplace, fuse)
       1515 ckpt, weight = torch_safe_load(weight)  # load ckpt
       1516 args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # combine model and default args, preferring model args
    -> 1517 model = (ckpt.get("ema") or ckpt["model"]).float()  # FP32 model
       1519 # Model compatibility updates
       1520 model.args = args  # attach args to model


    File ~/miniforge3/envs/ml-env/lib/python3.10/site-packages/torch/nn/modules/module.py:1186, in Module.float(self)
       1177 def float(self) -> Self:
       1178     r"""Casts all floating point parameters and buffers to ``float`` datatype.
       1179 
       1180     .. note::
       (...)
       1184         Module: self
       1185     """
    -> 1186     return self._apply(lambda t: t.float() if t.is_floating_point() else t)


    File ~/miniforge3/envs/ml-env/lib/python3.10/site-packages/ultralytics/nn/tasks.py:288, in BaseModel._apply(self, fn)
        279 def _apply(self, fn):
        280     """Apply a function to all tensors in the model, including Detect head attributes like stride and anchors.
        281 
        282     Args:
       (...)
        286         (BaseModel): An updated BaseModel object.
        287     """
    --> 288     self = super()._apply(fn)
        289     m = self.model[-1]  # Detect()
        290     if isinstance(
        291         m, Detect
        292     ):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect, YOLOEDetect, YOLOESegment


    File ~/miniforge3/envs/ml-env/lib/python3.10/site-packages/torch/nn/modules/module.py:933, in Module._apply(self, fn, recurse)
        931 if recurse:
        932     for module in self.children():
    --> 933         module._apply(fn)
        935 from torch._subclasses.fake_tensor import FakeTensor
        937 def compute_should_use_set_data(tensor, tensor_applied) -> bool:


    File ~/miniforge3/envs/ml-env/lib/python3.10/site-packages/torch/nn/modules/module.py:933, in Module._apply(self, fn, recurse)
        931 if recurse:
        932     for module in self.children():
    --> 933         module._apply(fn)
        935 from torch._subclasses.fake_tensor import FakeTensor
        937 def compute_should_use_set_data(tensor, tensor_applied) -> bool:


        [... skipping similar frames: Module._apply at line 933 (5 times)]


    File ~/miniforge3/envs/ml-env/lib/python3.10/site-packages/torch/nn/modules/module.py:933, in Module._apply(self, fn, recurse)
        931 if recurse:
        932     for module in self.children():
    --> 933         module._apply(fn)
        935 from torch._subclasses.fake_tensor import FakeTensor
        937 def compute_should_use_set_data(tensor, tensor_applied) -> bool:


    File ~/miniforge3/envs/ml-env/lib/python3.10/site-packages/torch/nn/modules/module.py:964, in Module._apply(self, fn, recurse)
        960 # Tensors stored in modules are graph leaves, and we don't want to
        961 # track autograd history of `param_applied`, so we have to use
        962 # `with torch.no_grad():`
        963 with torch.no_grad():
    --> 964     param_applied = fn(param)
        965 p_should_use_set_data = compute_should_use_set_data(param, param_applied)
        967 # subclasses may have multiple child tensors so we need to use swap_tensors


    File ~/miniforge3/envs/ml-env/lib/python3.10/site-packages/torch/nn/modules/module.py:1186, in Module.float.<locals>.<lambda>(t)
       1177 def float(self) -> Self:
       1178     r"""Casts all floating point parameters and buffers to ``float`` datatype.
       1179 
       1180     .. note::
       (...)
       1184         Module: self
       1185     """
    -> 1186     return self._apply(lambda t: t.float() if t.is_floating_point() else t)


    KeyboardInterrupt: 

