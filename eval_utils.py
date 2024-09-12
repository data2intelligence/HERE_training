# Copyright (c) OpenMMLab. All rights reserved.
# copied from mmclassification
import pdb
from numbers import Number

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import one_hot


def accuracy_numpy(pred, target, topk=(1,), thrs=0.):
    if isinstance(thrs, Number):
        thrs = (thrs,)
        res_single = True
    elif isinstance(thrs, tuple):
        res_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    res = []
    maxk = max(topk)
    num = pred.shape[0]

    static_inds = np.indices((num, maxk))[0]
    pred_label = pred.argpartition(-maxk, axis=1)[:, -maxk:]
    pred_score = pred[static_inds, pred_label]

    sort_inds = np.argsort(pred_score, axis=1)[:, ::-1]
    pred_label = pred_label[static_inds, sort_inds]
    pred_score = pred_score[static_inds, sort_inds]

    for k in topk:
        correct_k = pred_label[:, :k] == target.reshape(-1, 1)
        res_thr = []
        for thr in thrs:
            # Only prediction values larger than thr are counted as correct
            _correct_k = correct_k & (pred_score[:, :k] > thr)
            _correct_k = np.logical_or.reduce(_correct_k, axis=1)
            res_thr.append((_correct_k.sum() * 100. / num))
        if res_single:
            res.append(res_thr[0])
        else:
            res.append(res_thr)
    return res


def accuracy_torch(pred, target, topk=(1,), thrs=0.):
    if isinstance(thrs, Number):
        thrs = (thrs,)
        res_single = True
    elif isinstance(thrs, tuple):
        res_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    res = []
    maxk = max(topk)
    num = pred.size(0)
    pred = pred.float()
    pred_score, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
    for k in topk:
        res_thr = []
        for thr in thrs:
            # Only prediction values larger than thr are counted as correct
            _correct = correct & (pred_score.t() > thr)
            correct_k = _correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res_thr.append((correct_k.mul_(100. / num)))
        if res_single:
            res.append(res_thr[0])
        else:
            res.append(res_thr)
    return res


def accuracy(pred, target, topk=1, thrs=0.):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction.
        target (torch.Tensor | np.array): The target of each prediction
        topk (int | tuple[int]): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
        torch.Tensor | list[torch.Tensor] | list[list[torch.Tensor]]: Accuracy
            - torch.Tensor: If both ``topk`` and ``thrs`` is a single value.
            - list[torch.Tensor]: If one of ``topk`` or ``thrs`` is a tuple.
            - list[list[torch.Tensor]]: If both ``topk`` and ``thrs`` is a \
              tuple. And the first dim is ``topk``, the second dim is ``thrs``.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk,)
        return_single = True
    else:
        return_single = False

    assert isinstance(pred, (torch.Tensor, np.ndarray)), \
        f'The pred should be torch.Tensor or np.ndarray ' \
        f'instead of {type(pred)}.'
    assert isinstance(target, (torch.Tensor, np.ndarray)), \
        f'The target should be torch.Tensor or np.ndarray ' \
        f'instead of {type(target)}.'

    # torch version is faster in most situations.
    to_tensor = (lambda x: torch.from_numpy(x)
    if isinstance(x, np.ndarray) else x)
    pred = to_tensor(pred)
    target = to_tensor(target)

    res = accuracy_torch(pred, target, topk, thrs)

    return res[0] if return_single else res


class Accuracy(nn.Module):

    def __init__(self, topk=(1,)):
        """Module to calculate the accuracy.

        Args:
            topk (tuple): The criterion used to calculate the
                accuracy. Defaults to (1,).
        """
        super().__init__()
        self.topk = topk

    def forward(self, pred, target):
        """Forward function to calculate accuracy.

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
            list[torch.Tensor]: The accuracies under different topk criterions.
        """
        return accuracy(pred, target, self.topk)


def calculate_confusion_matrix(pred, target):
    """Calculate confusion matrix according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).

    Returns:
        torch.Tensor: Confusion matrix
            The shape is (C, C), where C is the number of classes.
    """

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    assert (
            isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor)), \
        (f'pred and target should be torch.Tensor or np.ndarray, '
         f'but got {type(pred)} and {type(target)}.')

    # Modified from PyTorch-Ignite
    num_classes = pred.size(1)
    pred_label = torch.argmax(pred, dim=1).flatten()
    target_label = target.flatten()
    assert len(pred_label) == len(target_label)

    with torch.no_grad():
        indices = num_classes * target_label + pred_label
        matrix = torch.bincount(indices, minlength=num_classes ** 2)
        matrix = matrix.reshape(num_classes, num_classes)
    return matrix


def precision_recall_f1(pred, target, average_mode='macro', thrs=0.):
    """Calculate precision, recall and f1 score according to the prediction and
    target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
        tuple: tuple containing precision, recall, f1 score.

            The type of precision, recall, f1 score is one of the following:

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """

    allowed_average_mode = ['macro', 'none']
    if average_mode not in allowed_average_mode:
        raise ValueError(f'Unsupport type of averaging {average_mode}.')

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    assert isinstance(pred, torch.Tensor), \
        (f'pred should be torch.Tensor or np.ndarray, but got {type(pred)}.')
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).long()
    assert isinstance(target, torch.Tensor), \
        f'target should be torch.Tensor or np.ndarray, ' \
        f'but got {type(target)}.'

    if isinstance(thrs, Number):
        thrs = (thrs,)
        return_single = True
    elif isinstance(thrs, tuple):
        return_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    num_classes = pred.size(1)
    pred_score, pred_label = torch.topk(pred, k=1)
    pred_score = pred_score.flatten()
    pred_label = pred_label.flatten()

    gt_positive = one_hot(target.flatten(), num_classes)

    precisions = []
    recalls = []
    f1_scores = []
    for thr in thrs:
        # Only prediction values larger than thr are counted as positive
        pred_positive = one_hot(pred_label, num_classes)
        if thr is not None:
            pred_positive[pred_score <= thr] = 0
        class_correct = (pred_positive & gt_positive).sum(0)
        precision = class_correct / np.maximum(pred_positive.sum(0), 1.) * 100
        recall = class_correct / np.maximum(gt_positive.sum(0), 1.) * 100
        f1_score = 2 * precision * recall / np.maximum(
            precision + recall,
            torch.finfo(torch.float32).eps)
        if average_mode == 'macro':
            precision = float(precision.mean())
            recall = float(recall.mean())
            f1_score = float(f1_score.mean())
        elif average_mode == 'none':
            precision = precision.detach().cpu().numpy()
            recall = recall.detach().cpu().numpy()
            f1_score = f1_score.detach().cpu().numpy()
        else:
            raise ValueError(f'Unsupport type of averaging {average_mode}.')
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    if return_single:
        return precisions[0], recalls[0], f1_scores[0]
    else:
        return precisions, recalls, f1_scores


def precision(pred, target, average_mode='macro', thrs=0.):
    """Calculate precision according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
         float | np.array | list[float | np.array]: Precision.

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """
    precisions, _, _ = precision_recall_f1(pred, target, average_mode, thrs)
    return precisions


def recall(pred, target, average_mode='macro', thrs=0.):
    """Calculate recall according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
         float | np.array | list[float | np.array]: Recall.

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """
    _, recalls, _ = precision_recall_f1(pred, target, average_mode, thrs)
    return recalls


def f1_score(pred, target, average_mode='macro', thrs=0.):
    """Calculate F1 score according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
         float | np.array | list[float | np.array]: F1 score.

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """
    _, _, f1_scores = precision_recall_f1(pred, target, average_mode, thrs)
    return f1_scores


def support(pred, target, average_mode='macro'):
    """Calculate the total number of occurrences of each label according to the
    prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted sum.
            Defaults to 'macro'.

    Returns:
        float | np.array: Support.

            - If the ``average_mode`` is set to macro, the function returns
              a single float.
            - If the ``average_mode`` is set to none, the function returns
              a np.array with shape C.
    """
    confusion_matrix = calculate_confusion_matrix(pred, target)
    with torch.no_grad():
        res = confusion_matrix.sum(1)
        if average_mode == 'macro':
            res = float(res.sum().numpy())
        elif average_mode == 'none':
            res = res.numpy()
        else:
            raise ValueError(f'Unsupport type of averaging {average_mode}.')
    return res


def mmcls_evaluate(gt_labels,
                   results,
                   metric='accuracy',
                   metric_options=None,
                   indices=None,
                   logger=None):
    """Evaluate the dataset.
    Args:
        gt_labels: Ground truth,
        results: Testing results of the dataset.
        metric (str | list[str]): Metrics to be evaluated.
            Default value is `accuracy`.
        metric_options (dict, optional): Options for calculating metrics.
            Allowed keys are 'topk', 'thrs' and 'average_mode'.
            Defaults to None.
        indices (list, optional): The indices of samples corresponding to
            the results. Defaults to None.
        logger (logging.Logger | str, optional): Logger used for printing
            related information during evaluation. Defaults to None.
    Returns:
        dict: evaluation results
    """
    if metric_options is None:
        metric_options = {'topk': (1, min(5, max(2, results.shape[1]//2)))}
    if isinstance(metric, str):
        metrics = [metric]
    else:
        metrics = metric
    allowed_metrics = [
        'accuracy', 'precision', 'recall', 'f1_score', 'support'
    ]
    eval_results = {}
    # results = np.vstack(results)

    if indices is not None:
        gt_labels = gt_labels[indices]
    num_imgs = len(results)
    assert len(gt_labels) == num_imgs, 'dataset testing results should ' \
                                       'be of the same length as gt_labels.'

    invalid_metrics = set(metrics) - set(allowed_metrics)
    if len(invalid_metrics) != 0:
        raise ValueError(f'metric {invalid_metrics} is not supported.')

    topk = metric_options.get('topk', (1, 5))
    thrs = metric_options.get('thrs')
    average_mode = metric_options.get('average_mode', 'macro')

    if 'accuracy' in metrics:
        if thrs is not None:
            acc = accuracy(results, gt_labels, topk=topk, thrs=thrs)
        else:
            acc = accuracy(results, gt_labels, topk=topk)
        if isinstance(topk, tuple):
            eval_results_ = {
                f'accuracy_top-{k}': a
                for k, a in zip(topk, acc)
            }
        else:
            eval_results_ = {'accuracy': acc}
        if isinstance(thrs, tuple):
            for key, values in eval_results_.items():
                eval_results.update({
                    f'{key}_thr_{thr:.2f}': value.item()
                    for thr, value in zip(thrs, values)
                })
        else:
            eval_results.update(
                {k: v.item()
                 for k, v in eval_results_.items()})

    if 'support' in metrics:
        support_value = support(
            results, gt_labels, average_mode=average_mode)
        eval_results['support'] = support_value

    precision_recall_f1_keys = ['precision', 'recall', 'f1_score']
    if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
        if thrs is not None:
            precision_recall_f1_values = precision_recall_f1(
                results, gt_labels, average_mode=average_mode, thrs=thrs)
        else:
            precision_recall_f1_values = precision_recall_f1(
                results, gt_labels, average_mode=average_mode)
        for key, values in zip(precision_recall_f1_keys,
                               precision_recall_f1_values):
            if key in metrics:
                if isinstance(thrs, tuple):
                    eval_results.update({
                        f'{key}_thr_{thr:.2f}': value
                        for thr, value in zip(thrs, values)
                    })
                else:
                    eval_results[key] = values

    return eval_results

