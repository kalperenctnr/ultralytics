from ultralytics import YOLO
from ultralytics.models.yolo.pose import PoseTrainerCustom
from ultralytics.models.yolo.pose import PoseTrainer
import torch

torch.cuda.empty_cache() 

# model = YOLO("yolo11s-pose.yaml")

model = YOLO("yolo11s-pose.yaml").load("/home/alperenc/ultralytics/yolo11s-pose.pt")  # build from YAML and transfer weights

# Freeze backbone (layers 0–10)
# for i in range(0, 10):  # 0 to 10 inclusive
#     for param in model.model.model[i].parameters():
#         param.requires_grad = False
    
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(f"{name}")
# torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
# for name, module in model.model.named_children():
#     print(name, "->", type(module))



# # # # Train the model
results = model.train(trainer=PoseTrainer,
                      data="/home/alperenc/Projects/PoseDatasets/Datasets/lm/PoseDatasetTrial/pose3d.yaml", 
                      cfg="/home/alperenc/ultralytics/train/config.yaml",
                      epochs=100, imgsz=640, val=True, augment=True, batch=32, pretrained=False, amp=True, workers=2, optimizer="Adam", close_mosaic=0, resume=False, fraction=1)