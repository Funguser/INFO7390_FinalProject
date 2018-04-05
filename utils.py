import os
import sys
import random
import time
from itertools import chain
from multiprocessing.pool import Pool

import pandas as pd
import numpy as np
from IPython.display import display
from keras.callbacks import Callback
from keras.preprocessing import image
import h5py

BASEPATH = os.environ.get('BASEPATH', '...')


def load_annotations():
    #match id with photos


def load_img_48(path):
    return image.load_img(path, target_size=(48, 48))


# def load_img_64(path):
#     return image.load_img(path, target_size=(64, 64))
#
#
# def load_img_80(path):
#     return image.load_img(path, target_size=(80, 80))


def load_img_96(path):
    return image.load_img(path, target_size=(96, 96))


#
# def load_img_128(path):
#     return image.load_img(path, target_size=(128, 128))
#
#
# def load_img_144(path):
#     return image.load_img(path, target_size=(144, 144))


# def load_img_160(path):
#     return image.load_img(path, target_size=(160, 160))

#
def load_img_176(path):
    return image.load_img(path, target_size=(176, 176))


# def load_img_192(path):
#     return image.load_img(path, target_size=(192, 192))
#
#
# def load_img_208(path):
#     return image.load_img(path, target_size=(208, 208))


def load_img_224(path):
    return image.load_img(path, target_size=(224, 224))


def load_X(annotations, dataset: str, size: int, worker=4):
    assert dataset in ('train', 'validation')
    print('Loading X_{}, size: {}, worker: {}'.format(dataset, size, worker))

    if dataset == 'validation':
        dirpath = os.path.join(BASEPATH, 'ai_challenger_scene_validation_20170908', 'full')
    else:
        dirpath = os.path.join(BASEPATH, 'ai_challenger_scene_train_20170904', 'full')
    filepaths = [os.path.join(dirpath, str(row['label_id']), row['image_id']) for _, row in annotations.iterrows()]

    func = globals().get(f'load_img_{size}')
    with Pool(worker) as p:
        images = p.map(func, filepaths)
        arrays = p.map(image.img_to_array, images)

    return np.array(arrays)


# def load_test(worker=6):
#     testpath = os.path.join(BASEPATH, 'ai_challenger_scene_test_a_20170922', 'scene_test_a_images_20170922')
#
#     image_ids = [one for one in os.listdir(testpath) if one.endswith('.jpg')]
#     image_paths = [os.path.join(testpath, one) for one in image_ids]
#
#     with Pool(worker) as p:
#         images = p.map(load_img_64, image_paths)
#         arrays = p.map(image.img_to_array, images)
#
#     return np.array(arrays), image_ids


def load_augmented_train(size: int = 48):
    def _load(img_fullpath):
        label_id = os.path.basename(os.path.dirname(img_fullpath))
        img = image.img_to_array(image.load_img(img_fullpath))
        return img, label_id

    dirpath = os.path.join(BASEPATH, 'ai_challenger_scene_train_20170904', 'augmented_{}'.format(size))
    img_paths = list(chain.from_iterable(((os.path.join(base, filename) for filename in filenames) for base, dirs, filenames in os.walk(dirpath))))

    m = len(img_paths)
    X_train, Y_train = np.zeros((m, size, size, 3), dtype=np.uint), []
    t0 = time.time()
    for i, p in enumerate(img_paths):
        if not i % 500:
            sys.stdout.write('{:.3f}s\t{}/{} = %{:.3f}\r'.format(time.time() - t0, i, m, i / m * 100))
            # for i in ids:
            #     p = img_paths[i]
        X, Y = _load(p)
        X_train[i] = X
        Y_train.append(int(Y))
    Y_train = np.array(pd.get_dummies(Y_train))

    return X_train, Y_train


def load_data_from_img(size: int = 48, annotations=None):
    if annotations == None:
        annotations_train, annotations_validation = load_annotations()
    else:
        annotations_train, annotations_validation = annotations

    X_train = load_X(annotations_train, 'train', size)
    Y_train = np.array(pd.get_dummies(annotations_train['label_id']))

    X_validation = load_X(annotations_validation, 'validation', size)
    Y_validation = np.array(pd.get_dummies(annotations_validation['label_id']))

    return X_train, Y_train, X_validation, Y_validation


def save_data_to_h5(X_train, Y_train, X_validation, Y_validation, size: int = 48):
    filename = './Data/dataset_{}.h5'.format(size)
    f = h5py.File(filename, 'w')
    print('Writing X_validation to {}'.format(filename))
    f.create_dataset('X_validation', data=X_validation, compression='lzf')
    print('Writing Y_validation to {}'.format(filename))
    f.create_dataset('Y_validation', data=Y_validation, compression='lzf')
    print('Writing X_train to {}'.format(filename))
    f.create_dataset('X_train', data=X_train, compression='lzf')
    print('Writing Y_train to {}'.format(filename))
    f.create_dataset('Y_train', data=Y_train, compression='lzf')
    f.close()


def load_data_from_h5(size: int = 48):
    f = h5py.File('./Data/dataset_{}.h5'.format(size), 'r')
    X_validation = np.array(f['X_validation'])
    Y_validation = np.array(f['Y_validation'])
    X_train = np.array(f['X_train'])
    Y_train = np.array(f['Y_train'])
    f.close()

    return X_train, Y_train, X_validation, Y_validation


def load_data(size: int = 48):
    if os.path.exists('./Data/dataset_{}.h5'.format(size)):
        return load_data_from_h5(size)
    else:
        data = load_data_from_img(size)
        save_data_to_h5(*data)
        return data


def load_partial_data(size: int = 48, classes: int = 10, use_stored: bool = True):
    if use_stored:
        f = h5py.File('./partial.h5', 'r')
        X_validation = np.array(f['X_validation'])
        Y_validation = np.array(f['Y_validation'])
        X_train = np.array(f['X_train'])
        Y_train = np.array(f['Y_train'])
        return X_train, Y_train, X_validation, Y_validation

    annotations_train, annotations_validation = load_annotations()
    label_ids = random.sample(list(range(80)), classes)
    annotations_train = annotations_train[annotations_train['label_id'].isin(label_ids)]
    annotations_validation = annotations_validation[annotations_validation['label_id'].isin(label_ids)]
    return load_data_from_img(size, (annotations_train, annotations_validation))


class Timer(Callback):
    def __init__(self, log_batch=True, log_epoch=True):
        super().__init__()
        self.log_batch = log_batch
        self.log_epoch = log_epoch

        self.start_time = self.end_time = 0

        self.epoch_start_times = None
        self.epoch_end_times = None

        self.batch_start_times = None
        self.batch_end_times = None

        self.current_epoch = 0
        self.step_per_batch = 0

    def set_params(self, params):
        super(Timer, self).set_params(params)

        self.epoch_start_times = np.zeros((params['epochs'],))
        self.epoch_end_times = np.zeros((params['epochs'],))

        self.step_per_batch = np.math.ceil(params['samples'] / params['batch_size'])
        self.batch_start_times = np.zeros((params['epochs'], self.step_per_batch))
        self.batch_end_times = np.zeros((params['epochs'], self.step_per_batch))

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        if self.log_epoch:
            total_epochs = self.params['epochs']
            sys.stdout.write(f'Epoch: {epoch+1}/{total_epochs}\n')
            sys.stdout.flush()
            self.epoch_start_times[epoch] = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if self.log_epoch:
            self.epoch_end_times[epoch] = time.time()

    def on_batch_begin(self, batch, logs=None):
        if self.log_batch:
            total_epochs = self.params['epochs']
            step = f'{batch+1}/{self.step_per_batch}'
            epoch = f'{self.current_epoch+1}/{total_epochs}'
            sys.stdout.write(f'Step: {step}\tEpoch: {epoch}\n')
            self.batch_start_times[self.current_epoch][batch] = time.time()

    def on_batch_end(self, batch, logs=None):
        if self.log_batch:
            self.batch_end_times[self.current_epoch][batch] = time.time()

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_train_end(self, logs=None):
        self.end_time = time.time()

    def summary(self):
        total_describe = pd.DataFrame({'Total Time': [self.end_time - self.start_time]}).describe()
        display(total_describe.T)

        epoch_time = self.epoch_end_times - self.epoch_start_times
        epoch_describe = pd.DataFrame(epoch_time, columns=['Epoch Times']).describe()
        display(epoch_describe.T)

        batch_time = self.batch_end_times - self.batch_start_times
        batch_describe = pd.DataFrame(batch_time.reshape(-1), columns=['Batch Times']).describe()
        display(batch_describe.T)
        return
