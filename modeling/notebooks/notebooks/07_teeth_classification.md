```python
from ultralytics import YOLO

# 1. Load your trained model
model_path = "../models/teeth classification/best.pt"
model = YOLO(model_path)

# 2. Run validation
# Note: Point this to your data.yaml file, which tells YOLO where the val images/labels are
data_yaml_path = "../data/teeth_classification/data.yaml"

print("Starting validation...")
metrics = model.val(data=data_yaml_path, split='val', device="mps")

# Print the results
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
```

    Starting validation...
    Ultralytics 8.4.37 🚀 Python-3.10.19 torch-2.10.0 MPS (Apple M1 Pro)
    YOLO26m-seg summary (fused): 149 layers, 23,509,010 parameters, 0 gradients, 131.2 GFLOPs
    [34m[1mval: [0mFast image access ✅ (ping: 0.0±0.0 ms, read: 1280.9±258.3 MB/s, size: 885.2 KB)
    [K[34m[1mval: [0mScanning /Users/nickol/My Fucking Stuff/machine learning/code/tooth_fairy/modeling/data/teeth_classification/val/labels... 478 images, 102 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 478/478 397.3it/s 1.2s.1s
    [34m[1mval: [0m/Users/nickol/My Fucking Stuff/machine learning/code/tooth_fairy/modeling/data/teeth_classification/val/images/train_481_q3.png: 1 duplicate labels removed
    [34m[1mval: [0mNew cache created: /Users/nickol/My Fucking Stuff/machine learning/code/tooth_fairy/modeling/data/teeth_classification/val/labels.cache
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Mask(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 30/30 5.8s/it 2:558.9sss
                       all        478        663      0.771      0.764      0.813      0.534       0.77      0.763      0.809      0.528
                  Impacted        121        121      0.872      0.926      0.939      0.579      0.872      0.926      0.939      0.584
                    Caries        330        542      0.669      0.601      0.687      0.489      0.667        0.6      0.679      0.471
    Speed: 1.2ms preprocess, 256.8ms inference, 0.0ms loss, 62.9ms postprocess per image
    Results saved to [1m/Users/nickol/My Fucking Stuff/machine learning/code/tooth_fairy/modeling/notebooks/runs/segment/val2[0m
    mAP50: 0.813
    mAP50-95: 0.534

