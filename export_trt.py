from ultralytics import YOLO

model = YOLO("runs/detect/train4/weights/best.pt")

model.export(format="engine", int8=True, device=0, data='datasets/batch.v2i.yolov11/data.yaml')