from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("./yolov8n.pt")
    result = model.train(data="your_path", epochs=100, imgsz=640, workers=8, device=0, batch=2, name='model_name')
