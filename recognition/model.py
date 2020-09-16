import cv2
import numpy as np
import mxnet as mx
# from PIL import Image, ImageFile
# from warnings import filterwarnings
from skimage import transform as trans

# ImageFile.LOAD_TRUNCATED_IMAGES = True
# filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

class Model:
    def __init__(self, weights_file, epoch=0, cuda=0, batch_size=1):
        print('loading', weights_file, epoch)
        if cuda == -1:
            ctx = mx.cpu()
        else:
            ctx = mx.gpu(cuda)
        sym, arg_params, aux_params = mx.model.load_checkpoint(weights_file, epoch)
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        image_size = (112, 112)
        self.image_size = image_size
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        model.bind(for_training=False, data_shapes=[('data', (batch_size, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        self.model = model
        src = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]], dtype=np.float32)
        self.src = src

    def preprocess(self, img, landmarks=None):

        landmarks = landmarks.reshape(5, 2)
        tform = trans.SimilarityTransform()
        tform.estimate(landmarks, self.src)
        M = tform.params[0:2, :]

        img = cv2.warpAffine(img, M, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))  # 3 x 112 x 112, RGB

        return np.array(img)

    def extract(self, images_batch):
        data = mx.nd.array(images_batch)
        data_batch = mx.io.DataBatch(data=(data,))
        self.model.forward(data_batch, is_train=False)
        features = self.model.get_outputs()[0].asnumpy()
        return features.astype(np.float32)


def init_default_model():
    import os
    root = os.path.dirname(os.path.abspath(__file__))
    model_weights_file = f'{root}/model/MobileFaceNet/model'
    model = Model(model_weights_file, 0, cuda=-1, batch_size=1)
    return model
