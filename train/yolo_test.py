from ultralytics import YOLO
import torch

# Clear CUDA memory
torch.cuda.empty_cache()

# Load the trained YOLOv8 pose model
model = YOLO("yolov8n-pose.yaml").load(
    "/home/alperenc/ultralytics/runs/pose/train21/weights/best.pt"
)

# Path to your dataset YAML file
data_yaml = "/home/alperenc/Projects/PoseDatasets/Datasets/lm/PoseDataset/000001/pose3d.yaml"  # make sure 'test:' is defined inside

# Run evaluation on test set
results = model.val(
    data=data_yaml,   # dataset config
    cfg="/home/alperenc/ultralytics/train/config.yaml",
    split="test",     # use test set from data.yaml
    save=True,        # save prediction images
    save_txt=True     # save results as .txt in YOLO format
)

# Print metrics
print(results)
