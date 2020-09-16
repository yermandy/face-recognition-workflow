import cv2, time
from skimage import transform as trans
import numpy as np
from detection import detection

# define a video capture object
capture = cv2.VideoCapture(0)

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
# capture.set(3, 200)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)
# capture.set(4, 200)

ret, frame = capture.read()

scale_percent = 40
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)

destination = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041]], dtype=np.float32)

destination[:, 0] -= 30
destination[:, 1] -= 51

center = np.array([[width / 2, height / 2]], dtype=np.float32)

destination += center


def capture_video():

    while True:

        ret, frame = capture.read()

        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        bboxes, landmarks = detection.detect(frame)

        if len(bboxes) > 0:
            bbox = bboxes[0].astype(np.int)
            lm = landmarks[0].astype(np.int)

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

            cv2.circle(frame, (lm[0], lm[1]), 1, (0, 0, 255), 4)
            cv2.circle(frame, (lm[2], lm[3]), 1, (0, 255, 255), 4)
            cv2.circle(frame, (lm[4], lm[5]), 1, (255, 0, 255), 4)
            cv2.circle(frame, (lm[6], lm[7]), 1, (0, 255, 0), 4)
            cv2.circle(frame, (lm[8], lm[9]), 1, (255, 0, 0), 4)

            source = np.array(lm, dtype=np.float32).reshape(5, 2)

            tform = trans.SimilarityTransform()
            tform.estimate(source, destination)
            M = tform.params[0:2, :]

            frame = cv2.warpAffine(frame, M, (width, height), borderValue=0.0)

        frame = cv2.flip(frame, 1)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_video()
