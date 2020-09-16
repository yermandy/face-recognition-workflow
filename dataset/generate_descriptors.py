import cv2
import os
import numpy as np

from recognition.extraction import feature_extraction
from recognition.model import init_default_model
from detection.detection import detect


def draw_landmarks(img, landmarks):
    cv2.circle(img, (landmarks[0], landmarks[1]), 1, (0, 0, 255), 4)
    cv2.circle(img, (landmarks[2], landmarks[3]), 1, (0, 255, 255), 4)
    cv2.circle(img, (landmarks[4], landmarks[5]), 1, (255, 0, 255), 4)
    cv2.circle(img, (landmarks[6], landmarks[7]), 1, (0, 255, 0), 4)
    cv2.circle(img, (landmarks[8], landmarks[9]), 1, (255, 0, 0), 4)


if __name__ == '__main__':

    model = init_default_model()

    root = os.path.dirname(os.path.abspath(__file__))
    identities_path = f'{root}/identities'
    folders = [folder for folder in os.listdir(identities_path) if not folder.startswith('.')]

    os.makedirs(f'{root}/descriptors', exist_ok=True)
    os.makedirs(f'{root}/detections', exist_ok=True)

    for folder in folders:

        folder_path = f'{identities_path}/{folder}'

        photos = [photo for photo in os.listdir(f'{folder_path}') if not photo.startswith('.')]

        if len(photos) == 0:
            continue

        descriptors = []

        for photo in photos:

            photo_path = f'{folder_path}/{photo}'
            img = cv2.imread(photo_path)
            box, landmarks = detect(img)

            if len(box) > 0:
                box = box[0].astype(int)
                landmarks = landmarks[0].astype(int)

                embeddings, norms = feature_extraction(model, img, landmarks)

                descriptors.append(embeddings)

                draw_landmarks(img, landmarks)
                cv2.imwrite(f'{root}/detections/{folder}_{photo}', img)

        identity_descriptor = np.average(descriptors, axis=0)

        identity_descriptor = identity_descriptor / np.linalg.norm(identity_descriptor)

        np.save(f'{root}/descriptors/{folder}.npy', identity_descriptor)


