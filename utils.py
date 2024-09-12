import os
import json
import cv2
import torch
import h5py
import numpy as np
import pandas as pd
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, f1_score, auc, roc_auc_score, roc_curve, classification_report, r2_score
from scipy.stats import percentileofscore, pearsonr, spearmanr
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('agg')
from PIL import Image, ImageDraw, ImageFont
from eval_utils import mmcls_evaluate


def save_hdf5(output_path, asset_dict, attr_dict=None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1,) + data_shape[1:]
            maxshape = (None,) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path

def new_web_annotation(cluster_label, min_dist, x, y, w, h, annoid_str, simple=False):
    anno = {
        "type": "Annotation",
        "body": [{
            "type": "TextualBody",
            "value": f"{cluster_label}" if simple else "{:d},({:.3f})".format(cluster_label, min_dist),
            "purpose": "tagging"
        }],
        "target": {
            "source": "http://localhost:3000/",
            "selector": {
                "type": "FragmentSelector",
                "conformsTo": "http://www.w3.org/TR/media-frags/",
                "value": f"xywh=pixel:{x},{y},{w},{h}"
            }
        },
        "@context": "http://www.w3.org/ns/anno.jsonld",
        "id": annoid_str
    }
    return anno


def get_svs_prefix(svs_filename):
    return os.path.basename(svs_filename).replace('.qptiff', '').replace('.ndpi', '').replace('.svs', '').replace('.tif', '')


def _assertLevelDownsamples(slide):
    level_downsamples = []
    dim_0 = slide.level_dimensions[0]

    for downsample, dim in zip(slide.level_downsamples, slide.level_dimensions):
        estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
        level_downsamples.append(estimated_downsample) if estimated_downsample != (
            downsample, downsample) else level_downsamples.append((downsample, downsample))

    return level_downsamples

def _assertLevelDownsamplesV2(level_dimensions, level_downsamples):
    level_downsamples_new = []
    dim_0 = level_dimensions[0]

    for downsample, dim in zip(level_downsamples, level_dimensions):
        estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
        level_downsamples_new.append(estimated_downsample) if estimated_downsample != (
            downsample, downsample) else level_downsamples_new.append((downsample, downsample))

    return level_downsamples_new



def generate_roc_curve_fixed(Y, Y_prob, task_name='HistoAnno',
                             label_names=['lob', 'duct', 'other'],
                             save_filename=None):
    Y0 = np.copy(Y)
    Y = np.eye(len(label_names))[Y]
    n_classes = Y.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    label_counts = []
    invalid_i = []

    thresholds = dict()
    optim_threshold_indices = []
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(Y[:, i], Y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        count = len(np.where(Y0 == i)[0])
        label_counts.append(count)
        if count == 0:
            invalid_i.append(i)
            optim_threshold_indices.append(-1)
        else:
            temp = tpr[i] - fpr[i]
            maxindex = temp.tolist().index(max(temp))
            optim_threshold_indices.append(maxindex)

    fpr["micro"], tpr["micro"], _ = roc_curve(Y.ravel(), Y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes) if i not in invalid_i]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        if i in invalid_i:
            continue
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= (n_classes - len(invalid_i))
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    lw = 2
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "olive", "fuchsia", "seagreen", "indigo",
                    "deepskyblue"])
    for i, color in zip(range(n_classes), colors):
        if i in invalid_i:
            continue
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC {0}:{1}({2}) ({3:0.2f})".format(i, label_names[i], label_counts[i], roc_auc[i]),
        )

        maxindex = optim_threshold_indices[i]
        optim_threshold = thresholds[i][maxindex]
        plt.text(fpr[i][maxindex],
                 tpr[i][maxindex],
                 '{:.3f}'.format(optim_threshold),
                 color=color,
                 fontsize='large')

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(task_name)
    plt.legend(loc="lower right")
    plt.grid()
    if save_filename is not None:
        plt.savefig(save_filename)
        plt.close()
    else:
        plt.show()



def softmax_stable(x):  # only 2-D
    x = np.exp(x - np.max(x, axis=1)[:, None])
    return x / x.sum(axis=1)[:, None]


class Accuracy_Logger(object):
    """Accuracy logger"""

    def __init__(self, n_classes, task_name="", label_names=None, ignore_label_ind=None):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.label_names = label_names
        self.task_name = task_name
        self.Y_hat = []
        self.Y = []
        self.Y_prob = []
        self.ignore_label_ind = ignore_label_ind
        self.fixed_ignore_label = False
        self.save_filename = None
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def log(self, Y_hat, Y, Y_prob):
        if isinstance(Y_hat, torch.Tensor):
            Y_hat_list = Y_hat.detach().cpu().numpy().tolist()
            if isinstance(Y, torch.Tensor):
                Y_list = Y.detach().cpu().numpy().tolist()
            elif isinstance(Y, np.ndarray):
                Y_list = Y.tolist()
            Y_prob = Y_prob.detach().cpu().numpy().tolist()
            if not isinstance(Y_hat_list, list):
                Y_hat_list = [Y_hat_list]
            if not isinstance(Y_list, list):
                Y_list = [Y_list]
            self.Y_hat += Y_hat_list
            self.Y += Y_list
            self.Y_prob += Y_prob
            for y_hat, y in zip(Y_hat_list, Y_list):
                self.data[y]["count"] += 1
                self.data[y]["correct"] += (y_hat == y)
        else:
            Y_hat = int(Y_hat)
            Y = int(Y)
            self.data[Y]["count"] += 1
            self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, count, correct, c):
        self.data[c]["count"] += count
        self.data[c]["correct"] += correct

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = 0
        else:
            acc = float(correct) / count

        return acc, correct, count

    def get_confusion_matrix(self):
        # self.correct_unseen_labels()
        self.fix_ignore_labels()
        cm = confusion_matrix(y_true=self.Y, y_pred=self.Y_hat, labels=np.arange(self.n_classes))
        return cm

    def get_f1_score(self, average='weighted'):
        # self.correct_unseen_labels()
        self.fix_ignore_labels()
        score = f1_score(y_true=self.Y, y_pred=self.Y_hat, average=average, labels=np.arange(self.n_classes),
                         zero_division=0)
        return score

    def fix_ignore_labels(self):
        if self.fixed_ignore_label or not (0 <= self.ignore_label_ind < self.n_classes):
            return

        if self.save_filename is not None:
            np.savetxt(self.save_filename.replace('.txt', '_Y_Original.txt'),
                       X=np.array(self.Y).reshape(-1), fmt='%d')
            np.savetxt(self.save_filename.replace('.txt', '_Y_hat_Original.txt'),
                       X=np.array(self.Y_hat).reshape(-1), fmt='%d')
            np.savetxt(self.save_filename.replace('.txt', '_Y_prob_Original.txt'),
                       X=np.array(self.Y_prob).reshape((-1, self.n_classes)), fmt='%.4f')

        Y = np.array(self.Y).reshape(-1)
        Y_hat = np.array(self.Y_hat).reshape(-1)
        Y_prob = np.array(self.Y_prob).reshape((-1, self.n_classes))
        Y_valid_inds = np.where(Y != self.ignore_label_ind)[0]
        Y = Y[Y_valid_inds]
        Y_hat = Y_hat[Y_valid_inds]
        Y_prob = np.delete(Y_prob[Y_valid_inds], self.ignore_label_ind, axis=1)
        Y_prob = softmax_stable(Y_prob)
        n_classes = self.n_classes - 1  # the ignore_label must be max(valid_labels) + 1
        label_names = [self.label_names[i] for i in range(len(self.label_names)) if i != self.ignore_label_ind]

        self.Y = Y
        self.Y_hat = Y_hat
        self.Y_prob = Y_prob
        self.n_classes = n_classes
        self.label_names = label_names
        self.fixed_ignore_label = True

    def get_auc_score(self, average='weighted'):
        # self.correct_unseen_labels()
        self.fix_ignore_labels()
        Y = np.array(self.Y).reshape(-1)
        Y_prob = np.array(self.Y_prob).reshape((-1, self.n_classes))
        if self.n_classes > 2:
            auc = roc_auc_score(y_true=Y, y_score=Y_prob, average=average, multi_class='ovo',
                                labels=np.arange(self.n_classes))
        else:
            auc = roc_auc_score(y_true=Y, y_score=Y_prob[:, 1], average=average, multi_class='ovo',
                                labels=np.arange(self.n_classes))

        return auc

    def get_roc_curve(self, save_filename):
        # self.correct_unseen_labels()
        self.fix_ignore_labels()
        Y = np.array(self.Y).reshape(-1)
        Y_prob = np.array(self.Y_prob).reshape((-1, self.n_classes))
        generate_roc_curve_fixed(Y, Y_prob, task_name=self.task_name,
                                 label_names=self.label_names,
                                 save_filename=save_filename)

    def get_classification_report(self, save_filename):
        self.fix_ignore_labels()
        Y = np.array(self.Y).reshape(-1)
        Y_prob = np.array(self.Y_prob).reshape((-1, self.n_classes))
        Y_hat = np.array(self.Y_hat).reshape(-1)
        eval_dict = mmcls_evaluate(Y, Y_prob,
                                   metric=['accuracy', 'precision', 'recall', 'f1_score', 'support'])
        target_names = ['class{}_{}'.format(ind, name) for ind, name in enumerate(self.label_names)]
        eval_dict.update(classification_report(Y, Y_hat, target_names=target_names, output_dict=True,
                                               labels=np.arange(self.n_classes)))
        with open(save_filename, 'w') as fp:
            json.dump(eval_dict, fp, indent=2)

    def correct_unseen_labels(self):
        self.fix_ignore_labels()
        Y = np.array(self.Y).reshape(-1)
        Y_prob = np.array(self.Y_prob).reshape((-1, self.n_classes))
        Y_labels = np.unique(Y)
        if len(Y_labels) != self.n_classes:
            unseen_labels = set([i for i in range(self.n_classes)]) - set(Y_labels)
            unseen_Y = list(unseen_labels)
            self.Y += unseen_Y
            self.Y_hat += unseen_Y
            self.Y_prob = np.concatenate([self.Y_prob,
                                          label_binarize(unseen_Y, classes=[i for i in range(len(self.label_names))])],
                                         axis=0)

    def save_data(self, save_filename=None):
        self.fix_ignore_labels()
        # self.correct_unseen_labels()
        np.savetxt(save_filename.replace('.txt', '_Y.txt'), X=np.array(self.Y), fmt='%d')
        np.savetxt(save_filename.replace('.txt', '_Y_hat.txt'), X=np.array(self.Y_hat), fmt='%d')
        np.savetxt(save_filename.replace('.txt', '_Y_prob.txt'), X=np.array(self.Y_prob), fmt='%.4f')

    def set_save_filename(self, save_filename=None):
        self.save_filename = save_filename



class Regression_Logger(object):
    def __init__(self):
        super(Regression_Logger, self).__init__()

        self.Y_hat = []
        self.Y = []

    def log(self, Y_hat, Y):
        if isinstance(Y_hat, torch.Tensor):
            Y_hat_list = Y_hat.detach().cpu().numpy().tolist()
            if isinstance(Y, torch.Tensor):
                Y_list = Y.detach().cpu().numpy().tolist()
            elif isinstance(Y, np.ndarray):
                Y_list = Y.tolist()
            if not isinstance(Y_hat_list, list):
                Y_hat_list = [Y_hat_list]
            if not isinstance(Y_list, list):
                Y_list = [Y_list]
            self.Y_hat += Y_hat_list
            self.Y += Y_list
        else:
            Y_hat = int(Y_hat)
            Y = int(Y)
            self.Y_hat.append(Y_hat)
            self.Y.append(Y)

    def mean_squared_error(self):
        if len(self.Y) == 0:
            return 0.
        else:
            Y_hat = np.array(self.Y_hat)
            Y = np.array(self.Y)
            return ((Y_hat - Y) ** 2).mean(axis=None)

    def compute_metrics(self):
        if len(self.Y) == 0:
            return {'r2score': 0.0,
                    'pearsonr_corr': 0.0,
                    'pearsonr_pvalue': 1.0,
                    'spearmanr_corr': 0.0,
                    'spearmanr_pvalue': 1.0}
        else:
            Y = np.array(self.Y).reshape(-1)
            Y_hat = np.array(self.Y_hat).reshape(-1)
            pearson_corr, pearsonr_pvalue = pearsonr(Y, Y_hat)
            spearmanr_corr, spearmanr_pvalue = spearmanr(Y, Y_hat)
            return {'r2score': r2_score(Y, Y_hat),
                    'pearsonr_corr': pearson_corr,
                    'pearsonr_pvalue': pearsonr_pvalue,
                    'spearmanr_corr': spearmanr_corr,
                    'spearmanr_pvalue': spearmanr_pvalue
                    }



def score2percentile(score, ref):
    percentile = percentileofscore(ref, score)
    return percentile


def to_percentiles(scores):
    from scipy.stats import rankdata
    scores = rankdata(scores, 'average') / len(scores) * 100
    return scores


def _assertLevelDownsamples(slide):
    level_downsamples = []
    dim_0 = slide.level_dimensions[0]

    for downsample, dim in zip(slide.level_downsamples, slide.level_dimensions):
        estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
        level_downsamples.append(estimated_downsample) if estimated_downsample != (
            downsample, downsample) else level_downsamples.append((downsample, downsample))

    return level_downsamples

def _assertLevelDownsamplesV2(level_dimensions, level_downsamples):
    level_downsamples_new = []
    dim_0 = level_dimensions[0]

    for downsample, dim in zip(level_downsamples, level_dimensions):
        estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))
        level_downsamples_new.append(estimated_downsample) if estimated_downsample != (
            downsample, downsample) else level_downsamples_new.append((downsample, downsample))

    return level_downsamples_new


def block_blending(slide, img, vis_level, top_left, bot_right, alpha=0.5, blank_canvas=False, block_size=1024):
    print('\ncomputing blend')
    level_downsamples = _assertLevelDownsamples(slide)
    downsample = level_downsamples[vis_level]
    w = img.shape[1]
    h = img.shape[0]
    block_size_x = min(block_size, w)
    block_size_y = min(block_size, h)
    print('using block size: {} x {}'.format(block_size_x, block_size_y))

    shift = top_left  # amount shifted w.r.t. (0,0)
    for x_start in range(top_left[0], bot_right[0], block_size_x * int(downsample[0])):
        for y_start in range(top_left[1], bot_right[1], block_size_y * int(downsample[1])):
            # print(x_start, y_start)

            # 1. convert wsi coordinates to image coordinates via shift and scale
            x_start_img = int((x_start - shift[0]) / int(downsample[0]))
            y_start_img = int((y_start - shift[1]) / int(downsample[1]))

            # 2. compute end points of blend tile, careful not to go over the edge of the image
            y_end_img = min(h, y_start_img + block_size_y)
            x_end_img = min(w, x_start_img + block_size_x)

            if y_end_img == y_start_img or x_end_img == x_start_img:
                continue
            # print('start_coord: {} end_coord: {}'.format((x_start_img, y_start_img), (x_end_img, y_end_img)))

            # 3. fetch blend block and size
            blend_block = img[y_start_img:y_end_img, x_start_img:x_end_img]
            blend_block_size = (x_end_img - x_start_img, y_end_img - y_start_img)

            if not blank_canvas:
                # 4. read actual wsi block as canvas block
                pt = (x_start, y_start)
                canvas = np.array(slide.read_region(pt, vis_level, blend_block_size).convert("RGB"))
            else:
                # 4. OR create blank canvas block
                canvas = np.array(Image.new(size=blend_block_size, mode="RGB", color=(255, 255, 255)))

            # 5. blend color block and canvas block
            img[y_start_img:y_end_img, x_start_img:x_end_img] = cv2.addWeighted(blend_block, alpha, canvas,
                                                                                1 - alpha, 0, canvas)
    return img


"""
  # downsample at which to visualize heatmap (-1 refers to downsample closest to 32x downsample)
  vis_level: 1
  # transparency for overlaying heatmap on background (0: background only, 1: foreground only)
  alpha: 0.4
  # whether to use a blank canvas instead of original slide
  blank_canvas: false
  # whether to also save the original H&E image
  save_orig: true
  # file extension for saving heatmap/original image
  save_ext: jpg
  # whether to calculate percentile scores in reference to the set of non-overlapping patches
  use_ref_scores: true
  # whether to use gaussian blur for further smoothing
  blur: false
  # whether to shift the 4 default corner points for checking if a patch is inside a foreground contour
  use_center_shift: true
  # whether to only compute heatmap for ROI specified by x1, x2, y1, y2
  use_roi: false 
  # whether to calculate heatmap with specified overlap (by default, coarse heatmap without overlap is always calculated)
  calc_heatmap: true
  # whether to binarize attention scores
  binarize: false
  # binarization threshold: (0, 1)
  binary_thresh: -1
  # factor for downscaling the heatmap before final dispaly
  custom_downsample: 1
  cmap: jet
"""


def visHeatmap(wsi, scores, coords, vis_level=1,
               top_left=None, bot_right=None,
               patch_size=(256, 256),
               blank_canvas=False, canvas_color=(220, 20, 50), alpha=0.4,
               blur=False, overlap=0.0,
               segment=True, use_holes=True,
               convert_to_percentiles=False,
               binarize=False, thresh=0.5,
               max_size=None,
               custom_downsample=1,
               cmap='jet'):
    """
    Args:
        scores (numpy array of float): Attention scores
        coords (numpy array of int, n_patches x 2): Corresponding coordinates (relative to lvl 0)
        vis_level (int): WSI pyramid level to visualize
        patch_size (tuple of int): Patch dimensions (relative to lvl 0)
        blank_canvas (bool): Whether to use a blank canvas to draw the heatmap (vs. using the original slide)
        canvas_color (tuple of uint8): Canvas color
        alpha (float [0, 1]): blending coefficient for overlaying heatmap onto original slide
        blur (bool): apply gaussian blurring
        overlap (float [0 1]): percentage of overlap between neighboring patches (only affect radius of blurring)
        segment (bool): whether to use tissue segmentation contour (must have already called self.segmentTissue such that
                        self.contours_tissue and self.holes_tissue are not None
        use_holes (bool): whether to also clip out detected tissue cavities (only in effect when segment == True)
        convert_to_percentiles (bool): whether to convert attention scores to percentiles
        binarize (bool): only display patches > threshold
        threshold (float): binarization threshold
        max_size (int): Maximum canvas size (clip if goes over)
        custom_downsample (int): additionally downscale the heatmap by specified factor
        cmap (str): name of matplotlib colormap to use
    """

    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)

    # downsample = self.level_downsamples[vis_level]
    level_downsamples = _assertLevelDownsamples(wsi)
    downsample = level_downsamples[vis_level]
    scale = [1 / downsample[0], 1 / downsample[1]]  # Scaling from 0 to desired level

    if len(scores.shape) == 2:
        scores = scores.flatten()

    if binarize:
        if thresh < 0:
            threshold = 1.0 / len(scores)

        else:
            threshold = thresh

    else:
        threshold = 0.0

    ##### calculate size of heatmap and filter coordinates/scores outside specified bbox region #####
    if top_left is not None and bot_right is not None:
        scores, coords = screen_coords(scores, coords, top_left, bot_right)
        coords = coords - top_left
        top_left = tuple(top_left)
        bot_right = tuple(bot_right)
        w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
        region_size = (w, h)

    else:
        # region_size = self.level_dim[vis_level]
        region_size = wsi.level_dimensions[vis_level]
        top_left = (0, 0)
        # bot_right = self.level_dim[0]
        bot_right = wsi.level_dimensions[0]
        w, h = region_size

    patch_size = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
    coords = np.ceil(coords * np.array(scale)).astype(int)

    print('\ncreating heatmap for: ')
    print('top_left: ', top_left, 'bot_right: ', bot_right)
    print('w: {}, h: {}'.format(w, h))
    print('scaled patch size: ', patch_size)

    ###### normalize filtered scores ######
    if convert_to_percentiles:
        scores = to_percentiles(scores)

    scores /= 100

    ######## calculate the heatmap of raw attention scores (before colormap)
    # by accumulating scores over overlapped regions ######

    # heatmap overlay: tracks attention score over each pixel of heatmap
    # overlay counter: tracks how many times attention score is accumulated over each pixel of heatmap
    overlay = np.full(np.flip(region_size), 0).astype(float)
    counter = np.full(np.flip(region_size), 0).astype(np.uint16)
    count = 0
    for idx in range(len(coords)):
        score = scores[idx]
        coord = coords[idx]
        if score >= threshold:
            if binarize:
                score = 1.0
                count += 1
        else:
            score = 0.0
        # accumulate attention
        overlay[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] += score
        # accumulate counter
        counter[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] += 1

    if binarize:
        print('\nbinarized tiles based on cutoff of {}'.format(threshold))
        print('identified {}/{} patches as positive'.format(count, len(coords)))

    # fetch attended region and average accumulated attention
    zero_mask = counter == 0

    if binarize:
        overlay[~zero_mask] = np.around(overlay[~zero_mask] / counter[~zero_mask])
    else:
        overlay[~zero_mask] = overlay[~zero_mask] / counter[~zero_mask]
    del counter
    if blur:
        overlay = cv2.GaussianBlur(overlay, tuple((patch_size * (1 - overlap)).astype(int) * 2 + 1), 0)

    # if segment:
    #     tissue_mask = self.get_seg_mask(region_size, scale, use_holes=use_holes, offset=tuple(top_left))
    #     # return Image.fromarray(tissue_mask) # tissue mask

    if not blank_canvas:
        # downsample original image and use as canvas
        img = np.array(wsi.read_region(top_left, vis_level, region_size).convert("RGB"))
    else:
        # use blank canvas
        img = np.array(Image.new(size=region_size, mode="RGB", color=(255, 255, 255)))

        # return Image.fromarray(img) #raw image

    print('\ncomputing heatmap image')
    print('total of {} patches'.format(len(coords)))
    twenty_percent_chunk = max(1, int(len(coords) * 0.2))

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    for idx in range(len(coords)):
        if (idx + 1) % twenty_percent_chunk == 0:
            print('progress: {}/{}'.format(idx, len(coords)))

        score = scores[idx]
        coord = coords[idx]
        if score >= threshold:
            # attention block
            raw_block = overlay[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]]

            # # image block (either blank canvas or orig image)
            # img_block = img[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]].copy()
            #
            # # color block (cmap applied to attention block)
            # color_block = (cmap(raw_block) * 255)[:, :, :3].astype(np.uint8)
            #
            # if segment:
            #     # tissue mask block
            #     mask_block = tissue_mask[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]]
            #     # copy over only tissue masked portion of color block
            #     img_block[mask_block] = color_block[mask_block]
            # else:
            #     # copy over entire color block
            #     img_block = color_block

            img_block = (cmap(raw_block) * 255)[:, :, :3].astype(np.uint8)

            # rewrite image block
            img[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] = img_block.copy()

    # return Image.fromarray(img) #overlay
    print('Done')
    del overlay

    if blur:
        img = cv2.GaussianBlur(img, tuple((patch_size * (1 - overlap)).astype(int) * 2 + 1), 0)

    if alpha < 1.0:
        img = block_blending(wsi, img, vis_level, top_left, bot_right, alpha=alpha, blank_canvas=blank_canvas,
                             block_size=1024)

    img = Image.fromarray(img)
    w, h = img.size

    if custom_downsample > 1:
        img = img.resize((int(w / custom_downsample), int(h / custom_downsample)))

    if max_size is not None and (w > max_size or h > max_size):
        resizeFactor = max_size / w if w > h else max_size / h
        img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))

    return img


def visWSI(wsi, vis_level=0, color=(0, 255, 0), hole_color=(0, 0, 255), annot_color=(255, 0, 0),
           line_thickness=250, max_size=None, top_left=None, bot_right=None, custom_downsample=1,
           view_slide_only=False,
           number_contours=False, seg_display=True, annot_display=True):
    level_downsamples = _assertLevelDownsamples(wsi)
    downsample = level_downsamples[vis_level]
    scale = [1 / downsample[0], 1 / downsample[1]]

    if top_left is not None and bot_right is not None:
        top_left = tuple(top_left)
        bot_right = tuple(bot_right)
        w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
        region_size = (w, h)
    else:
        top_left = (0, 0)
        region_size = wsi.level_dimensions[vis_level]

    img = wsi.read_region(top_left, vis_level, region_size).convert("RGB")
    w, h = img.size
    if custom_downsample > 1:
        img = img.resize((int(w / custom_downsample), int(h / custom_downsample)))

    if max_size is not None and (w > max_size or h > max_size):
        resizeFactor = max_size / w if w > h else max_size / h
        img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))

    return img

