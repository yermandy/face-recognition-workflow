import cv2
from detection.detection import detect

def init_video():

    capture = cv2.VideoCapture(0)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
    capture.set(3, 200)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)
    capture.set(4, 200)

    ret, frame = capture.read()
    scale_percent = 40
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    return capture, dim

def capture_video():

    capture, dim = init_video()

    while True:

        ret, frame = capture.read()

        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        bboxes, landmarks = detect(frame)

        if len(bboxes) > 0:
            bbox = bboxes[0].astype(int)
            lm = landmarks[0].astype(int)

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

            cv2.circle(frame, (lm[0], lm[1]), 1, (0, 0, 255), 4)
            cv2.circle(frame, (lm[2], lm[3]), 1, (0, 255, 255), 4)
            cv2.circle(frame, (lm[4], lm[5]), 1, (255, 0, 255), 4)
            cv2.circle(frame, (lm[6], lm[7]), 1, (0, 255, 0), 4)
            cv2.circle(frame, (lm[8], lm[9]), 1, (255, 0, 0), 4)

        frame = cv2.flip(frame, 1)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    capture_video()
