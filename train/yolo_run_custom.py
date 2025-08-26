from ultralytics import YOLO
import torch


torch.cuda.empty_cache() 

model = YOLO("yolo11s-pose.yaml").load("/home/alperenc/ultralytics/yolo11s-pose.pt")  # build from YAML and transfer weights

# # Freeze backbone (layers 0–10)
# for i in range(0, 10):  # 0 to 10 inclusive
#     for param in model.model.model[i].parameters():
#         param.requires_grad = False
    
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}")

# Train the model
# results = model.train(data="/home/alperenc/Projects/PoseDatasets/Datasets/lm/PoseDataset/pose3d.yaml", 
#                       cfg="/home/alperenc/ultralytics/train/config.yaml",
#                       epochs=1000, imgsz=640, val=True, augment=True, batch=8, pretrained=True)