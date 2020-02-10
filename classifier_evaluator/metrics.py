# -*- coding: utf-8 -*-
"""
Created on 2018-07-01, Last Modified on 2020-02-10;
@author Yi.Q
"""
"""
Metrics to assess the performance on a classification model
Measures are from: https://en.wikipedia.org/wiki/Confusion_matrix;

———————————————————————————————————————————————————————————————————————————
         |                                       actual condition          |
         |—————————————————————————————————————————————————————————————————
         |                    |    actual positive    |  actual negative   |
         |—————————————————————————————————————————————————————————————————
Predicted| predicted positive |     true positive     |   false positive   |
condition|—————————————————————————————————————————————————————————————————
         | predicted negative |     false negative    |   true negative    |
—————————|—————————————————————————————————————————————————————————————————

Terms and calculations:
    thresholds: pre-set thresholds on which the follower metrics are calculated;
    TP: true positive, power, the number of obs. that is conditioned positive and predicted positive;
    FN: false negative, type II error, the number of obs. that is conditioned positive and predicted negative;
    FP: false positive, type I error, the number of obs. that is conditioned negative and predicted positive;
    TN: true negative, the number of obs. that is conditioned negative and predicted negative;

    Recall: TPR, probability of detecting true positive condition, = TP / (TP + FN);
    FNR: miss rate, = FN / (TP + FN);
    FPR: probability of false alarm, = FP / (FP + TN);
    TNR: tue negative rate, = TN / (FP + TN);

    Precision: Positive predictive value(PPV), = TP / (TP + FP);
    FDR: false discovery rate, FP / (TP + FP);
    FOR: false omission rate, FN / (FN + TN);
    NPV: negative predictive value, TN / (FN + TN);

    prevalence: = condition positive / total population;
    Accuracy: = (TP + TN) / (TP + FN + FP + TN);
    LR+: = TPR / FPR;
    LR-: = FNR / TNR;
    DOR: diagnostic odds ratio, = LR+ / LR-;
    F1: 2 / ((1/Recall) + (1/Precision));
"""
import numpy as np
from typing import Union, Tuple, Optional

from classifier_evaluator.styles.format_confusion_matrix_by_prob_df import convert_confusion_matrix_by_prob_to_table_with_reformat_precision

_FULL_METRICS = ['TP', 'FN', 'FP', 'TN',
                 'Recall', 'FNR', 'FPR', 'TNR', 'Precision', 'FOR', 'FDR', 'NPV',
                 'Prevalence', 'Accuracy', 'LR+', 'LR-', 'DOR', 'F1']

_DEFAULT_POS_LABEL = True


def accuracy_rate(true: np.ndarray, predicted: np.ndarray) -> float:
    """
    calculate accuracy rate (ACC):
        ( true positive + true negative ) / sample size

    :param true: numpy.ndarray(shape=(m), ), an array of true classes;
    :param predicted: numpy.ndarray(shape=(m), ), an array of predicted classes;
    :return: float, fraction, accuracy rate
    """
    sample_size = len(true)
    return (predicted == true).sum() / sample_size


def accuracy_rate_by_prob(true: np.ndarray, predicted_prob: np.ndarray,
                          threshold: float, pos_label: Union[str, bool, int] = _DEFAULT_POS_LABEL) -> float:
    """
    accuracy rate for binary classification according to a given threshold;

    :param true: numpy.ndarray(shape=(m), ), an array of true classes;
    :param predicted_prob: numpy.ndarray(shape=(m), ),
        an array of predicted probabilities of being the positive class;
    :param threshold: float, [0, 1], the threshold set on predicted probabilities
        such that any predicted probability greater or equal to the threshold will be classified as the positive class,
        e.g., predicted_class(i | predicted_prob_i >= threshold) = positive;
    :param pos_label: [str, bool, int], positive class label, label that is considered as the positive class;
    :return: float, fraction, accuracy rate
    """
    # convert true series to positive - negative class
    true = (true == pos_label)
    predicted = (predicted_prob >= threshold)

    return accuracy_rate(true=true, predicted=predicted)


def _condition_counter(true: np.ndarray,
                       predicted: np.ndarray,
                       true_condition: Union[bool, int, str, list, tuple, set],
                       predicted_condition: Union[bool, int, str, list, tuple, set]) -> int:
    """
    count the number of instances such that their true value being true_condition and predicted being predicted_condition;
    :param true: numpy.ndarray(shape=(m), ), an array of true classes;
    :param predicted: numpy.ndarray(shape=(m), ), an array of predicted classes;
    :param true_condition: [bool, int, str, list, tuple, set], the true condition(s) that is being count;
    :param predicted_condition: [bool, int, str, list, tuple, set], the predicted condition(s) that is being count;
    :return: total count of the instance with the joined condition, int;
    """
    if isinstance(true_condition, (str, bool, int)):
        true_condition = [true_condition]
    elif isinstance(true_condition, set):
        true_condition = list(true_condition)
    elif isinstance(true_condition, (list, tuple)):
        pass
    else:
        raise TypeError(f"Type of true conditions {type(true_condition)} is not acceptable.")

    if isinstance(predicted_condition, (str, bool, int)):
        predicted_condition = [predicted_condition]
    elif isinstance(predicted_condition, set):
        predicted_condition = list(predicted_condition)
    elif isinstance(predicted_condition, (list, tuple)):
        pass
    else:
        raise TypeError(f"Type of true conditions {type(predicted_condition)} is not acceptable.")

    return (np.isin(true, true_condition) & np.isin(predicted, predicted_condition)).sum()


def prevalence(series: np.ndarray,
               condition: Union[bool, int, str, list, tuple, set],
               rate: bool = False) -> Union[int, float]:
    """
    count the number of instances with (one of) the given condition(s);
    :param series: numpy.ndarray(shape=(m), ) containing different conditions;
    :param condition: [bool, int, str, list, tuple, set], a selected condition;
    :param rate: bool, return prevalence rate instead of count;
    :return: the count of the instances with the condition(s);
    """
    if isinstance(condition, (str, bool, int)):
        condition = [condition]
    elif isinstance(condition, set):
        condition = list(condition)
    elif isinstance(condition, (list, tuple)):
        pass
    else:
        raise TypeError(f"Type of true conditions {type(condition)} is not acceptable.")

    if rate:
        return np.isin(series, condition).sum() / series.size if series.size > 0 else 0
    else:
        return np.isin(series, condition).sum()


def recall_rate(true: np.ndarray,
                predicted: np.ndarray,
                condition: Union[bool, int, str, list, tuple, set]) -> float:
    """
    compute recall rate, refer module description for calculation;
    :param true: numpy.ndarray(shape=(m), ), an array of true classes;
    :param predicted: numpy.ndarray(shape=(m), ), an array of predicted classes;
    :param condition: [bool, int, str, list, tuple, set], the true condition(s) that is being count;
    :return: recall rate, float;
    """
    joint_condition_count = _condition_counter(true=true,
                                               predicted=predicted,
                                               true_condition=condition,
                                               predicted_condition=condition)

    prevalence_count = prevalence(series=true,
                                  condition=condition,
                                  rate=False)

    recall = joint_condition_count / prevalence_count if prevalence_count > 0 else 0

    return recall


def recall_rate_by_prob(true: np.ndarray,
                        predicted_prob: np.ndarray,
                        threshold: float = 0.5,
                        pos_label: Union[bool, int, str] = _DEFAULT_POS_LABEL) -> float:
    """
    compute recall rate when predicted prob is given, refer module description for calculation;
    :param true: numpy.ndarray(shape=(m), ), an array of true classes;
    :param predicted_prob: numpy.ndarray(shape=(m), ), an array of predicted probability of being positive classe;
    :param threshold: float, the threshold over which predicted probabilities will be counted as positive;
    :param pos_label: [bool, int, str], the true condition(s) that is being count;
    :return: recall rate, float;
    """

    true = true == pos_label
    predicted = predicted_prob >= threshold

    recall = recall_rate(true=true,
                         predicted=predicted,
                         condition=True)

    return recall


def precision_rate(true: np.ndarray,
                   predicted: np.ndarray,
                   condition: Union[bool, int, str, list, tuple, set]) -> float:
    """
    compute precision rate, refer module description for calculation;
    :param true: numpy.ndarray(shape=(m), ), an array of true classes;
    :param predicted: numpy.ndarray(shape=(m), ), an array of predicted classes;
    :param condition: [bool, int, str, list, tuple, set], the true condition(s) that is being count;
    :return: precision rate, float;
    """
    joint_condition_count = _condition_counter(true=true,
                                               predicted=predicted,
                                               true_condition=condition,
                                               predicted_condition=condition)

    prevalence_count = prevalence(series=predicted,
                                  condition=condition,
                                  rate=False)

    precision = joint_condition_count / prevalence_count if prevalence_count > 0 else 0

    return precision


def precision_rate_by_prob(true: np.ndarray,
                           predicted_prob: np.ndarray,
                           threshold: float = 0.5,
                           pos_label: Union[bool, int, str] = _DEFAULT_POS_LABEL) -> float:
    """
    compute precision rate when predicted prob is given, refer module description for calculation;
    :param true: numpy.ndarray(shape=(m), ), an array of true classes;
    :param predicted_prob: numpy.ndarray(shape=(m), ), an array of predicted probability of being positive classe;
    :param threshold: float, the threshold over which predicted probabilities will be counted as positive;
    :param pos_label: [bool, int, str], the true condition(s) that is being count;
    :return: precision rate, float;
    """

    true = true == pos_label
    predicted = predicted_prob >= threshold

    precision = precision_rate(true=true,
                               predicted=predicted,
                               condition=True)

    return precision


def confusion_matrix(true: np.ndarray, predicted: np.ndarray, normalize: Union[bool, str] = False) -> Union[tuple, dict]:
    """
    compute confusion matrix

        terms: pair condition: joined condition of both observed condition and predicted condition, e.g., (True, False);
               true condition: observed condition of an instance, e.g., True;
               predicted condition: predicted condition of an instance, e.g., True;
    :param true: true: numpy.ndarray(shape=(m), ), an array of true classes;
    :param predicted: numpy.ndarray(shape=(m), ), an array of predicted classes;
    :param normalize: [bool, 'true', 'predicted'], if the confusion matrix needs to be normalised:
        True: default, normalized along true classes; False: not normalized;
        'true': normalized along true classes; 'predicted': normalized along predicted classes;
        e.g., normalisation along true classes: normalized_condition (for i) = count(predicted = i, true = i) / (true = i)
    :return: dict, a confusion matrix:
        key: tuple, (true_class, predicted_class);
        value: [int, float], the number of instances of that pair condition (true_class, predicted_class), or
            normalisation of the number of the instances of that pair condition;
    """
    labels = [label.item() for label in set(true).union(set(predicted))]  # .item convert numpy dtypes to native types;

    confusion_matrix_dict = dict()
    for true_label in labels:
        for predicted_label in labels:
            confusion_matrix_dict[(true_label, predicted_label)] = \
                _condition_counter(true=true, predicted=predicted,
                                   true_condition=true_label, predicted_condition=predicted_label)

    if normalize:
        normalize_factor = 0 if normalize != 'predicted' else 1
        normalized_confusion_matrix_dict = normalize_confusion_matrix(confusion_matrix_dict=confusion_matrix_dict,
                                                                      normalize_index=normalize_factor)
        return confusion_matrix_dict, normalized_confusion_matrix_dict
    else:
        return confusion_matrix_dict


def normalize_confusion_matrix(confusion_matrix_dict: dict, normalize_index: int = 0) -> dict:
    """
    normalize a confusion matrix;

    :param confusion_matrix_dict: dict, a confusion matrix, e.g.,
        {('a', 'a'): 1, ('a', 'b'): 2, ('b', 'a'): 3, ('b', 'b'): 4};
    :param normalize_index: int, [0, 1], the position index of the confusion matrix dict key (tuple) where normalization
    is performed along;
    :return: a normalized confusion matrix;
    """
    labels = set([key[0] for key in confusion_matrix_dict.keys()])
    normalized_confusion_matrix_dict = dict()

    sum_dict = {label: 0 for label in labels}
    for condition, condition_count in confusion_matrix_dict.items():
        sum_dict[condition[normalize_index]] += condition_count

    for condition, condition_count in confusion_matrix_dict.items():
        normalized_confusion_matrix_dict[condition] = \
            condition_count / sum_dict[condition[normalize_index]] if sum_dict[condition[normalize_index]] else 0

    return normalized_confusion_matrix_dict


def confusion_matrix_by_prob(true: np.ndarray,
                             predicted_prob: np.ndarray,
                             thresholds: Optional[Union[list, tuple, np.ndarray]] = None,
                             pos_label: Union[bool, str, int] = _DEFAULT_POS_LABEL,
                             output_metrics: Optional[list] = None,
                             table: bool = True,
                             **kwargs):
    """
    confusion matrix for binary classification according to a given set of thresholds;

    :param true: numpy.ndarray(shape=(m), ), an array of true classes;
    :param predicted_prob: numpy.ndarray(shape=(m), ),
        an array of predicted probabilities of being the positive class;
    :param thresholds: [list, tuple, np.array, None] the thresholds set on predicted probabilities
        such that any predicted probability greater or equal to the threshold will be classified as the positive class;
    :param pos_label: [str, bool, int], positive class label, label that is considered as the positive class;
    :param output_metrics: [list, None], metrics to be outputted if selected;
    :param table: bool, if exported as a pd table table;
    :return: dict, a set of confusion matrices, {threshold: {metric_name: metric_value, ...}, ...};
    """
    # convert true series to positive series
    true = true == pos_label

    # select output:
    if isinstance(output_metrics, list):
        for selected_metric in output_metrics:
            if selected_metric not in _FULL_METRICS:
                raise KeyError(f"metric {selected_metric} is not recognized.")
    elif output_metrics == 'confusion':
        output_metrics = ['TP', 'FN', 'FP', 'TN',
                          'Recall', 'FNR', 'FPR', 'TNR', 'Precision', 'FOR', 'FDR', 'NPV',
                          'Prevalence', 'Accuracy']
    else:
        output_metrics = _FULL_METRICS

    metrics_by_thresholds = dict()
    for threshold in thresholds:
        metrics_by_threshold = dict()
        predicted = predicted_prob >= threshold
        confusion_matrix_dict = confusion_matrix(true=true, predicted=predicted, normalize=False)

        confusion_matrix_nor_true = normalize_confusion_matrix(confusion_matrix_dict=confusion_matrix_dict,
                                                               normalize_index=0)

        confusion_matrix_nor_predicted = normalize_confusion_matrix(confusion_matrix_dict=confusion_matrix_dict,
                                                                    normalize_index=1)

        if 'TP' in output_metrics:
            metrics_by_threshold['TP'] = confusion_matrix_dict[(True, True)]

        if 'FN' in output_metrics:
            metrics_by_threshold['FN'] = confusion_matrix_dict[(True, False)]

        if 'FP' in output_metrics:
            metrics_by_threshold['FP'] = confusion_matrix_dict[(False, True)]

        if 'TN' in output_metrics:
            metrics_by_threshold['TN'] = confusion_matrix_dict[(False, False)]

        if 'Recall' in output_metrics:
            metrics_by_threshold['Recall'] = confusion_matrix_nor_true[(True, True)]

        if 'FNR' in output_metrics:
            metrics_by_threshold['FNR'] = confusion_matrix_nor_true[(True, False)]

        if 'FPR' in output_metrics:
            metrics_by_threshold['FPR'] = confusion_matrix_nor_true[(False, True)]

        if 'TNR' in output_metrics:
            metrics_by_threshold['TNR'] = confusion_matrix_nor_true[(False, False)]

        if 'Precision' in output_metrics:
            metrics_by_threshold['Precision'] = confusion_matrix_nor_predicted[(True, True)]

        if 'FOR' in output_metrics:
            metrics_by_threshold['FOR'] = confusion_matrix_nor_predicted[(True, False)]

        if 'FDR' in output_metrics:
            metrics_by_threshold['FDR'] = confusion_matrix_nor_predicted[(False, True)]

        if 'NPV' in output_metrics:
            metrics_by_threshold['NPV'] = confusion_matrix_nor_predicted[(False, False)]

        if 'Prevalence' in output_metrics:
            metrics_by_threshold['Prevalence'] = \
                (confusion_matrix_dict[(True, True)] + confusion_matrix_dict[(True, False)]) / sum(confusion_matrix_dict.values())

        if 'Accuracy' in output_metrics:
            metrics_by_threshold['Accuracy'] = \
                (confusion_matrix_dict[(True, True)] + confusion_matrix_dict[(False, False)]) / sum(confusion_matrix_dict.values())

        if 'LR+' in output_metrics:
            # positive likelihood ratio:
            try:
                metrics_by_threshold['LR+'] = confusion_matrix_nor_true[(True, True)] / confusion_matrix_nor_true[(False, True)]
            except ZeroDivisionError:
                metrics_by_threshold['LR+'] = '-'

        if 'LR-' in output_metrics:
            # negative likelihood ratio:
            try:
                metrics_by_threshold['LR-'] = confusion_matrix_nor_true[(True, False)] / confusion_matrix_nor_true[(False, False)]
            except ZeroDivisionError:
                metrics_by_threshold['LR-'] = '-'

        if 'DOR' in output_metrics:
            # diagnostic odds ratio:
            try:
                metrics_by_threshold['DOR'] = (confusion_matrix_nor_true[(True, True)] / confusion_matrix_nor_true[(False, True)]) / \
                                              (confusion_matrix_nor_true[(True, False)] / confusion_matrix_nor_true[(False, False)])
            except ZeroDivisionError:
                metrics_by_threshold['DOR'] = '-'

        if 'F1' in output_metrics:
            # F1 score:
            try:
                metrics_by_threshold['F1'] = 2 * (confusion_matrix_nor_true[(True, True)] * confusion_matrix_nor_predicted[(True, True)]) / \
                                             (confusion_matrix_nor_true[(True, True)] + confusion_matrix_nor_predicted[(True, True)])
            except ZeroDivisionError:
                metrics_by_threshold['F1'] = '-'

        metrics_by_thresholds[threshold] = metrics_by_threshold

    if table:
        if 'metric_order' in kwargs:
            metric_order = kwargs['metric_order']
        else:
            metric_order = None
        metrics_by_thresholds = \
            convert_confusion_matrix_by_prob_to_table_with_reformat_precision(metrics_by_thresholds=metrics_by_thresholds,
                                                                              metric_order=metric_order)

    return metrics_by_thresholds


def roc(true: np.ndarray, predicted_prob: np.ndarray, pos_label: Union[bool, str, int] = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Receiver operating characteristic (ROC):

    :param true: numpy.ndarray(shape=(m), ), an array of true classes;
    :param predicted_prob: numpy.ndarray(shape=(m), ),
        an array of predicted probabilities of being the positive class;
    :param pos_label: [str, bool, int], positive class label, label that is considered as the positive class;
    :return: tuple, (thresholds, fpr, tpr), thresholds: numpy.ndarray, fpr: numpy.ndarray, tpr: numpy.ndarray;
    """
    true = true == pos_label

    # sort by predicted prob:
    sorted_index = np.argsort(predicted_prob)[::-1]
    true = true[sorted_index]
    predicted_prob = predicted_prob[sorted_index]

    tp = 0
    fp = 0
    condition_positive = true.sum()
    condition_negative = (1 - true).sum()

    # set thresholds at predicted_prob:
    thresholds = predicted_prob
    tpr, fpr = [], []
    for i in range(len(thresholds)):
        if true[i] == 1:
            tp += 1
        elif true[i] == 0:
            fp += 1

        tpr_i = tp / condition_positive  # consistent with confusion matrix by prob;
        fpr_i = fp / condition_negative

        tpr.append(tpr_i)
        fpr.append(fpr_i)

    tpr, fpr = np.array(tpr), np.array(fpr)

    return fpr, tpr, thresholds


def roc_auc(**kwargs) -> float:
    """
    compute AUC (area under the curve):
    :param kwargs:
        set 1:
        fpr: numpy.ndarray(shape=(m), ), false positive rate;
        tpr: numpy.ndarray(shape=(m), ), true positive rate;
        set 2:
        true: numpy.ndarray(shape=(m), ), an array of true classes;
        predicted_prob: numpy.ndarray(shape=(m), ),
            an array of predicted probabilities of being the positive class;
        pos_label: [str, bool, int], positive class label, label that is considered as the positive class;
    :return: auc, float, area under the curve;
    """
    if ('fpr' in kwargs) and ('tpr' in kwargs):
        fpr = kwargs['fpr']
        tpr = kwargs['tpr']
    elif ('true' in kwargs) and ('predicted_prob' in kwargs):
        # if fpr and tpr are not given: re-compute roc:
        if 'pos_label' in kwargs:
            pos_label = kwargs['pos_label']
        else:
            pos_label = _DEFAULT_POS_LABEL

        fpr, tpr, thresholds = roc(true=kwargs['true'], predicted_prob=kwargs['predicted_prob'], pos_label=pos_label)
    else:
        raise KeyError(f"given arguments are not sufficient to compute AUC.")

    fpr, tpr = np.insert(fpr, 0, 1), np.insert(tpr, 0, 1)  # add 1 to the first position of the array;
    fpr, tpr = np.append(fpr, 0), np.append(tpr, 0)  # add 0 to the last position of the arrays;

    auc = float(np.sum(1 / 2 * (tpr[0:-1] + tpr[1:]) * (- np.diff(fpr))))

    return auc


def roc_margins(**kwargs) -> dict:
    """
    compute differences between a fpr and tpr:
        also computes
    :param kwargs:
        set 1:
        fpr: numpy.ndarray(shape=(m), ), false positive rate;
        tpr: numpy.ndarray(shape=(m), ), true positive rate;
        set 2:
        true: numpy.ndarray(shape=(m), ), an array of true classes;
        predicted_prob: numpy.ndarray(shape=(m), ),
            an array of predicted probabilities of being the positive class;
        pos_label: [str, bool, int], positive class label, label that is considered as the positive class;
    :return: roc margin dict, dict, including:
        fpr: false positive rate, numpy.ndarray(shape=(m), );
        tpr, true positive rate, numpy.ndarray(shape=(m), );
        margins: margin curve, numpy.ndarray(shape=(m), );
        max_margin: the maximum margin between fpr and tpr;
        max_margin_threshold: threshold for such maximum margin;
    """

    if ('fpr' in kwargs) and ('tpr' in kwargs) and ('thresholds' in kwargs):
        fpr = kwargs['fpr']
        tpr = kwargs['tpr']
        thresholds = kwargs['thresholds']
    elif ('true' in kwargs) and ('predicted_prob' in kwargs):
        # if fpr and tpr are not given: re-compute roc:
        if 'pos_label' in kwargs:
            pos_label = kwargs['pos_label']
        else:
            pos_label = _DEFAULT_POS_LABEL

        fpr, tpr, thresholds = roc(true=kwargs['true'], predicted_prob=kwargs['predicted_prob'], pos_label=pos_label)
    else:
        raise KeyError(f"given arguments are not sufficient to compute AUC.")

    margins = tpr - fpr  # the margin between tpr and fpr
    max_margin = margins.max()
    max_margin_threshold = thresholds[margins.argmax()]

    roc_margin_dict = {'fpr': fpr,
                       'tpr': tpr,
                       'thresholds': thresholds,
                       'margins': margins,
                       'max_margin': max_margin,
                       'max_margin_threshold': max_margin_threshold}

    return roc_margin_dict
