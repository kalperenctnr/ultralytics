from ultralytics import YOLO
import torch

# Clear CUDA memory
torch.cuda.empty_cache()

# Load the trained YOLOv8 pose model
model = YOLO("/home/alperenc/ultralytics/runs/pose/train21/weights/best.pt")

# Path to the single image
image_path = "/home/alperenc/Projects/PoseDatasets/Datasets/lm/PoseDataset/000001/images/test/000042.png"

# Run prediction
results = model.predict(
    source=image_path,  # path to the image
    save=True,          # save the image with predictions
    save_txt=True,      # save keypoints in YOLO txt format
    imgsz=640,          # input size
    device='cuda'       # 'cuda' or 'cpu'
)

# Print the results object
print(results)
