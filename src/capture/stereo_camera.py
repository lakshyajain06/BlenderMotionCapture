import cv2 as cv


class StereoCamera:
    def __init__(self, resolution):
        # define camera 1
        self.capture1 = cv.VideoCapture(0)
        self.capture1.set(cv.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.capture1.set(cv.CAP_PROP_FRAME_HEIGHT, resolution[1])

        # define camera 2
        self.capture2 = cv.VideoCapture(1)
        self.capture2.set(cv.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.capture2.set(cv.CAP_PROP_FRAME_HEIGHT, resolution[1])

    def __del__(self):
        self.capture1.release()
        self.capture2.release()
        cv.destroyAllWindows()
    
    def get_frame(self, BGR=True):
        isTrue1, frame1 = self.capture1.read()
        isTrue2, frame2 = self.capture2.read()

        # convert from BGR to RGB if BGR is False
        if not BGR:
            frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
            frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2RGB)

        return isTrue1 and isTrue2, frame1, frame2
    
