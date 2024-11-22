from threading import Thread
from ultralytics import YOLO
from ultralytics import solutions
import torch, cv2

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VideoDetect:

    def __init__(self, frame=None):
        self.frame = frame
        self.img = None
        self.stopped = False
        # Define region points
        region_points = [(140, 180), (520, 180), (520, 400), (140, 400)]
        # Init Object Counter
        self.counter = solutions.ObjectCounter(show=True, region=region_points, model="runs/detect/train/weights/best.pt")

    def start(self):
        self.stopped = False
        Thread(target=self.thread_safe_predict, args=()).start()
        return self

    def thread_safe_predict(self):
        while not self.stopped:
            if self.frame is None:
                break
            self.img = self.frame.copy()
            print(2)
            cv2.imshow('frame', self.img)
            cv2.waitKey(1)
            print(3)
            # img = self.counter.count(img)
                    
    def stop(self):
        self.stopped = True
        