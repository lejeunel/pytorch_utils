import numpy as np
from skimage import io
import glob
import os
from sklearn.metrics import (f1_score, roc_curve, auc, precision_recall_curve)
import torch
import shutil
import progressbar as pbar
import scipy.misc as scm
from pytorch_utils import im_utils as ptimu


def save_checkpoint(state,
                    is_best,
                    path,
                    fname_cp='checkpoint.pth.tar',
                    fname_bm='best_model.pth.tar'):

    cp_path = os.path.join(path, fname_cp)
    bm_path = os.path.join(path, fname_bm)

    if (not os.path.exists(path)):
        os.makedirs(path)

    torch.save(state, cp_path)

    if (is_best):
        shutil.copyfile(cp_path, bm_path)


def load_checkpoint(path, model, gpu=False):

    if (gpu):
        checkpoint = torch.load(
            path, map_location=lambda storage, loc: storage)
    else:
        # checkpoint = torch.load(path)
        checkpoint = torch.load(
            path, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])

    return model


def extract_unet_feats(dl, model):

    model.eval()

    with pbar(max_val=len(dl)) as bar:
        for i, data in enumerate(dl):
            f_ = model.get_bottom_layer(data.image)


def comp_unet_input_shape(shape, depth, max_shape=None):

    img_height, img_width, _ = shape

    if(max_shape is not None):
        if (img_height > max_shape[0]):
            img_height = max_shape[0]

        if (img_width > max_shape[1]):
            img_width = max_shape[1]

    div_var = 2**depth
    # check if we can achieve the last layer without ending up in decimal size
    if (float(shape[1]) / div_var) % 1 != 0:
        # back-calculation of image width
        img_width = int(int(float(shape[1]) / div_var) * div_var)
        # self.logger.info('Cannot take image width, use ' +
        #                     str(img_width) + ' instead')

    return (img_height, img_width)


def read_csv(csvName, seqStart=None, seqEnd=None):
    out = np.loadtxt(
        open(csvName, "rb"), delimiter=";", skiprows=5)[seqStart:seqEnd, :]
    if ((seqStart is not None) or (seqEnd is not None)):
        out[:, 0] = np.arange(0, seqEnd - seqStart)
    return out


def get_truth_frames(path):

    frame_path = os.path.join(path, 'results')
    if (not os.path.exists(frame_path)):
        print('Couldnt find {}. Generating frames'.format(frame_path))
        os.makedirs(frame_path)
        res = np.load(os.path.join(path, 'results.npz'))['ksp_scores_mat']
        for i in range(res.shape[-1]):
            io.imsave(
                os.path.join(frame_path, 'im_{0:04d}.png'.format(i)),
                res[..., i] * 255)

    frames = sorted(glob.glob(os.path.join(frame_path, 'im_*.png')))
    frames = [f for f in frames if ('pb' not in f)]

    return frames


def get_all_scores(y, y_pred, n_points):

    fpr, tpr, _ = roc_curve(y, y_pred)
    auc_ = auc(np.asarray(fpr).ravel(), np.asarray(tpr).ravel())
    pr, rc, _ = precision_recall_curve(y, y_pred)
    probas_thr = np.linspace(0, 1, n_points)
    f1 = 2 * (pr * rc) / (pr + rc)

    return (fpr, tpr, auc_, pr, rc, f1, probas_thr)


def center_pred(pred):
    max_ = np.max(pred)
    min_ = np.min(pred)

    return (pred - min_) * (max_ - min_)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def predict(model, im_paths, in_shape=(100, 100), batch_size=4):
    im_chunks = chunks(im_paths, batch_size)
    preds = []
    for i, ims in enumerate(im_chunks):
        print('predicting {}/{}'.format(i + 1, len(im_paths) // batch_size))
        ims = [io.imread(im).astype(float) / 255 for im in ims]
        ims = [
            ptimu.to_tensor((io.imread(im).astype(float) / 255)).unsqueeze(0)
            for im in ims
        ]
        ims = torch.cat(ims)
        pred = model(ims)

