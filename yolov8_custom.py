from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('best.pt')

# Run inference on the source

results = model(source=0, show=True, conf=0.8, save=True) # generator of Results objects
