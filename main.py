import cv2
import os
import json
import numpy as np

from detection.detection import detect
from recognition.extraction import feature_extraction
from recognition.model import init_default_model


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


def load_identities():
    root = os.getcwd()
    descriptors_path = f'{root}/dataset/descriptors'
    identities = {f.split('.')[0]: np.load(f'{descriptors_path}/{f}') for f in os.listdir(descriptors_path)}
    return identities


def capture_video():
    capture, dim = init_video()

    model = init_default_model()

    identities = load_identities()

    descriptors = np.array(list(identities.values()))

    descriptors_ids = list(identities.keys())

    with open(f'{os.getcwd()}/dataset/identity_name.json') as f:
        identity_names = json.load(f)

    while True:

        ret, frame = capture.read()

        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        frame = cv2.flip(frame, 1)

        bboxes, landmarks = detect(frame)

        if len(bboxes) > 0:
            bbox = bboxes[0].astype(np.int)
            landmarks = landmarks[0].astype(np.int)

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)

            cv2.circle(frame, (landmarks[0], landmarks[1]), 1, (0, 0, 255), 2)
            cv2.circle(frame, (landmarks[2], landmarks[3]), 1, (0, 255, 255), 2)
            cv2.circle(frame, (landmarks[4], landmarks[5]), 1, (255, 0, 255), 2)
            cv2.circle(frame, (landmarks[6], landmarks[7]), 1, (0, 255, 0), 2)
            cv2.circle(frame, (landmarks[8], landmarks[9]), 1, (255, 0, 0), 2)

            embeddings, norms = feature_extraction(model, frame, landmarks)

            distances = 1 - (descriptors @ embeddings)

            argmin_dist = np.argmin(distances)

            threshold = 0.5

            if distances[argmin_dist] > threshold:
                identity_name = 'unknown'
            else:
                identity_id = descriptors_ids[np.argmin(distances)]
                identity_name = identity_names[identity_id]

            # print(identity_names[identity_id])
            # print(distances)

            cv2.putText(frame, f'{identity_name}', (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        # Display the resulting frame 
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_video()
