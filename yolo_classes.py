from ultralytics import YOLO

model = YOLO('C:/Users/khale/LIA/train_yolo11x/weights/best_yolo11x.pt')  # or use your custom trained model path

print(model.names)