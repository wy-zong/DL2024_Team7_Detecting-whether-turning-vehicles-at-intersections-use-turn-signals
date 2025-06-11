from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("./yolov8n.pt")
    result = model.train(data=r"D:\深度學習概論_熊博安\期末專題\wy\final project v3.v1i.yolov8\data.yaml", epochs=100, imgsz=640, workers=8, device=0, batch=2, name=r'D:\深度學習概論_熊博安\期末專題\wy\wy_custom_v3_model')