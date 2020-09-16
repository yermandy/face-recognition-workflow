import numpy as np
import argparse
from .model import Model


def parse_args():
    parser = argparse.ArgumentParser(description='ArcFace feature extraction')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--workers', default=8, type=int, help='Workers number')
    parser.add_argument('--dataset', required=True, type=str, help='Dataset name')
    parser.add_argument('--ref', required=True, type=str, help='Path to the file with paths and bounding boxes')
    parser.add_argument('--bb_scale', default=0.5, type=int, help='Bounding box scale')
    parser.add_argument('--cuda', default=-1, type=int, help='Cuda device')
    parser.add_argument('--model', default='model/MS1MV2-ResNet100-Arcface/model', help='path to load model.')
    return parser.parse_args()


def batches(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


def feature_extraction(model, image, landmarks):
    """
    Extracts feature vectors from photos using landmarks
    
    Parameters
    ----------
    model
        Model to use for feature extraction
    image : np.array
        Image with face in it, shape: (3, 112, 112)
    landmarks : np.array
        Landmarks in image, shape: (10,)
    """

    images_batch = np.empty((1, 3, 112, 112))

    # For now, let's process one face at a time
    images_batch[0] = model.preprocess(image, landmarks=landmarks)

    embeddings = model.extract(images_batch)

    norms = np.linalg.norm(embeddings)
    embeddings = embeddings / norms

    return embeddings[0], norms

if __name__ == '__main__':
    args = parse_args()
    model = Model(args.model, 0, args.cuda, args.batch_size)
    # feature_extraction(model, image, landmarks)
