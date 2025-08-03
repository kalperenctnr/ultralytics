from ultralytics import YOLO
import torch

torch.cuda.empty_cache() 
# # Load a model
model = YOLO("yolo11n-pose.yaml")  # build a new model from YAML
model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n-pose.yaml").load("yolo11n-pose.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="/home/alperenc/Projects/PoseDatasets/Datasets/lm/PoseDataset/000001/pose3d.yaml", 
                      cfg="/home/alperenc/ultralytics/train/config.yaml",
                      epochs=1000, imgsz=640, val=True, augment=True, batch=4)