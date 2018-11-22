import os
from pytorch_utils import im_utils as utls
import torch
import numpy as np
from skimage import io
from torchvision import transforms as trfms
from torchvision.transforms import Lambda
import matplotlib.pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from collections import namedtuple
from torch.utils import data

Data = namedtuple('Data', ('image', 'obj_prior', 'truth', 'im_orig'))



def normalize_keypoints_func(keypoints_on_images, random_state, parents,
                             hooks):
    return keypoints_on_images


class MetaDataloader():
    """
    This wraps several Dataloaders
    """

    def __init__(self, dataloaders, mode='train'):

        self.dataloaders = dataloaders

        self.mode = mode

        for d in dataloaders:
            d.mode = mode

        lens = [(0, len(d)) for d in self.dataloaders]
        self.idx_bins = list()

        for i, l in enumerate(lens):
            if (i == 0):
                self.idx_bins.append(l)
            else:
                self.idx_bins.append((self.idx_bins[i - 1][1],
                                      self.idx_bins[i - 1][1] + l[1]))

    def __len__(self):
        return sum([len(dl) for dl in self.dataloaders])

    def get_classes_weights(self):

        weights_ = []
        weights_dict = dict()
        for d in self.dataloaders:
            weights_.append(d.get_classes_weights())

        classes = np.unique([list(w.keys()) for w in weights_])
        weights_dict = {c: [] for c in classes}
        for w in weights_:
            for c in w.keys():
                weights_dict[c].append(w[c])

        for c in weights_dict.keys():
            weights_dict[c] = np.mean(weights_dict[c])

        return weights_dict

    def __getitem__(self, idx):

        # Find which dataloader has idx
        idx_loader = np.where(
            [(idx >= b[0]) & (idx < b[1]) for b in self.idx_bins])[0][0]
        dl = self.dataloaders[idx_loader]

        return dl[idx - self.idx_bins[idx_loader][0]]

    def sample_uniform(self):

        random_dl = np.random.choice(np.arange(len(self.dataloaders)))

        return self.dataloaders[random_dl].sample_uniform()


class Dataset(data.Dataset):
    def __init__(
            self,
            in_shape,
            im_paths=[],
            truth_paths=None,
            locs2d=None,
            sig_prior=0.1,
            mode='train',
            cuda=False,
            normalize=False,  # compute mean and stdev of dataset
            sometimes_rate=0.5,
            seed=0):

        self.im_paths = im_paths

        self.scale = iaa.Scale(in_shape)
        self.sometimes = lambda aug: iaa.Sometimes(sometimes_rate, aug)

        self.aug_affines = self.sometimes(
            iaa.Noop()
        )

        self.aug_noise = iaa.AdditiveGaussianNoise(
                        scale=0.1)

        self.truth_paths = truth_paths
        self.locs2d = locs2d
        self.sig_prior = sig_prior
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.in_shape = in_shape
        self.seed = 0

        if (normalize):
            self.mode = 'eval'
            self.mean, self.std = self.comp_normalization_factors()
            self.normalize = utls.NormalizeAug(mean=self.mean, std=self.std)
        else: # Do scaling
            self.normalize = utls.PixelScaling()
            
        self.mode = mode

    def set_aug_affine(self, affine):
        self.aug_affines = self.sometimes(affine)

    def set_aug_noise(self, noise):
        self.aug_noise = self.sometimes(noise)

    def comp_normalization_factors(self):

        # im = io.imread(im_path)
        ims = [io.imread(p) for p in self.im_paths]
        ims = np.asarray(ims)
        mean = [np.mean(ims[..., a]) for a in range(ims.shape[-1])]
        std = [np.std(ims[..., a]) for a in range(ims.shape[-1])]

        return mean, std


    def add_imgs(self, im_paths):
        self.im_paths += im_paths

    def add_truths(self, truth_paths):
        self.truth_paths += truth_paths

    def __len__(self):
        return len(self.im_paths)

    def sample_uniform(self, n=1):

        random_sample_idx = np.random.choice(
            np.arange(len(self.im_paths)),
            size=n,
            replace=False)

        samples = [self.__getitem__(r) for r in random_sample_idx]
        batch = Data(*zip(*samples))

        return Data(torch.cat(batch.image),
                    torch.cat(batch.obj_prior),
                    torch.cat(batch.truth),
                    np.asarray(batch.im_orig))

    def get_classes_weights(self):
        truths = np.asarray([io.imread(p) // 255 for p in self.truth_paths])
        n_pix = np.prod(truths.shape)

        classes = sorted(np.unique(truths))
        weights = {
            c: 1 - (np.sum(truths.ravel() == c) / n_pix)
            for c in classes
        }

        return weights

    def __getitem__(self, idx):

        im_path = self.im_paths[idx]

        # When truths are resized, the values change
        # we apply a threshold to get back to binary

        im = io.imread(im_path)

        # remove alpha channel if exists
        if (im.shape[-1] > 3):
            im = im[..., 0:3]

        im_orig = im.copy()
        im_shape = im.shape[0:2]

        truth = None
        obj_prior = None
        aug_affines_det = self.aug_affines.to_deterministic()

        if (self.mode == 'train'):
            if (self.truth_paths is None):
                raise Exception('mode is train but no truth paths given')

            truth_path = self.truth_paths[idx]

            truth = io.imread(truth_path).astype(np.uint8)

            truth = self.scale.augment_image(truth)
            if (self.mode == 'train'):
                truth = aug_affines_det.augment_images([truth])[0]

            truth = truth / 255

            if (len(truth.shape) == 3):
                truth = np.mean(truth, axis=-1)
                truth = utls.to_tensor(truth[..., None]).to(
                    self.device)
            elif (len(truth.shape) == 2):
                truth = utls.to_tensor(truth[..., None]).to(
                    self.device)
            elif (len(truth.shape) == 4):
                truth = utls.to_tensor(truth[..., 0, None]).to(
                    self.device)
        else:
            truth = torch.empty(im_shape).to(self.device)

        # Apply data augmentation
        if(self.locs2d is not None):
            locs = self.locs2d[self.locs2d[:, 0] == idx, 3:].tolist()
            locs = [utls.coord2Pixel(x, y, self.in_shape[1], self.in_shape[0])
                    for x, y in locs]

            keypoints = ia.KeypointsOnImage(
                [ia.Keypoint(x=l[1], y=l[0]) for l in locs], shape=self.in_shape)
            keypoints = self.scale.augment_keypoints([keypoints])[0]
            if (self.mode == 'train'):
                keypoints = aug_affines_det.augment_keypoints([keypoints])[0]

            if (len(locs) > 0):
                obj_prior = [
                    utls.make_2d_gauss(self.in_shape,
                                       self.sig_prior * max(self.in_shape),
                                       (kp.y, kp.x)) for kp in keypoints.keypoints
                ]
                obj_prior = np.asarray(obj_prior).sum(axis=0)[..., None]
                obj_prior += obj_prior.min()
                obj_prior /= obj_prior.max()
            else:
                obj_prior = (
                    np.ones(self.in_shape) / np.prod(self.in_shape))[..., None]

            obj_prior = utls.to_tensor(obj_prior).to(self.device)
        else:
            obj_prior = torch.empty(im_shape).to(self.device)

        if (self.mode == 'train'):
            im = aug_affines_det.augment_images([im])[0]
            im = self.aug_noise.augment_images([im])[0]
        im = self.scale.augment_image(im)

        im = self.normalize.augment_image(im)

        # Move to device
        im = utls.to_tensor(im).to(self.device)

        return im, obj_prior, truth, im_orig
