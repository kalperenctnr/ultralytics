from ultralytics import YOLO
import torch

torch.cuda.empty_cache() 

model = YOLO("yolo11s-pose.yaml").load("/home/alperenc/ultralytics/runs/pose/train20/weights/best.pt")  # build from YAML and transfer weights

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
results = model.train(data="/home/alperenc/Projects/PoseDatasets/Datasets/lm/PoseDataset/pose3d.yaml", 
                      cfg="/home/alperenc/ultralytics/train/config.yaml",
                      epochs=10, imgsz=640, val=True, augment=True, batch=16, pretrained=False, amp=False)