from threading import Thread, Lock
import cv2, time

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src = 0, width = 640, height = 480) :
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self) :
        if self.started :
            print("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            time.sleep(0.01)
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self) :
        if self.grabbed:
            self.read_lock.acquire()
            frame = self.frame.copy()
            frame = cv2.resize(frame, (640, 640))
            self.read_lock.release()
            return frame
        else:
            return None

    def stop(self) :
        self.started = False
        self.stream.release()
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()