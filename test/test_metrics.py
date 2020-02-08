import numpy as np
from classifier_evaluator.metrics import accuracy_rate, accuracy_rate_by_prob
from classifier_evaluator.metrics import prevalence
from classifier_evaluator.metrics import recall_rate, recall_rate_by_prob
from classifier_evaluator.metrics import precision_rate, precision_rate_by_prob
from classifier_evaluator.metrics import confusion_matrix, confusion_matrix_by_prob
from classifier_evaluator.metrics import roc


TRUE_SERIES = np.array(['A', 'A', 'A', 'B', 'B', 'C', 'C', 'D', 'E'])
PREDICTED_SERIES = np.array(['A', 'A', 'B', 'B', 'D', 'C', 'C', 'C', 'F'])


def test_accuracy_rate():
    actual_accuracy_rate = accuracy_rate(true=TRUE_SERIES,
                                         predicted=PREDICTED_SERIES)

    expected_accuracy_rate = 5 / 9

    assert abs(expected_accuracy_rate - actual_accuracy_rate) < 0.0001


def test_prevalence():
    actual_count = prevalence(series=TRUE_SERIES,
                              condition='A',
                              rate=False)

    expected_count = 3

    assert actual_count == expected_count

    actual_rate = prevalence(series=TRUE_SERIES,
                             condition='A',
                             rate=True)

    expected_rate = 1 / 3

    assert abs(expected_rate - actual_rate) < 0.0001

    actual_rate_group = prevalence(series=TRUE_SERIES,
                                   condition={'A', 'B'},
                                   rate=True)

    expected_rate_group = 5 / 9

    assert abs(expected_rate_group - actual_rate_group) < 0.0001


def test_recall_rate():
    actual_rate = recall_rate(true=TRUE_SERIES,
                              predicted=PREDICTED_SERIES,
                              condition='A')

    expected_rate = 2 / 3

    assert abs(expected_rate - actual_rate) < 0.0001

    # when condition is a set:
    actual_rate_group = recall_rate(true=TRUE_SERIES,
                                    predicted=PREDICTED_SERIES,
                                    condition={'A', 'B'})

    expected_rate_group = 4 / 5

    assert abs(expected_rate_group - actual_rate_group) < 0.0001


def test_precision_rate():
    actual_rate = precision_rate(true=TRUE_SERIES,
                                 predicted=PREDICTED_SERIES,
                                 condition='A')

    expected_rate = 1

    assert abs(expected_rate - actual_rate) < 0.0001

    # when condition is a set:
    actual_rate_group = precision_rate(true=TRUE_SERIES,
                                       predicted=PREDICTED_SERIES,
                                       condition={'A', 'B'})

    expected_rate_group = 1

    assert abs(expected_rate_group - actual_rate_group) < 0.0001


def test_confusion_matrix():
    actual_confusion_dict = confusion_matrix(true=TRUE_SERIES, predicted=PREDICTED_SERIES, normalize=False)
    expected_confusion_dict = {('A', 'A'): 2,
                               ('A', 'B'): 1,
                               ('A', 'C'): 0,
                               ('A', 'D'): 0,
                               ('A', 'E'): 0,
                               ('A', 'F'): 0,

                               ('B', 'A'): 0,
                               ('B', 'B'): 1,
                               ('B', 'C'): 0,
                               ('B', 'D'): 1,
                               ('B', 'E'): 0,
                               ('B', 'F'): 0,

                               ('C', 'A'): 0,
                               ('C', 'B'): 0,
                               ('C', 'C'): 2,
                               ('C', 'D'): 0,
                               ('C', 'E'): 0,
                               ('C', 'F'): 0,

                               ('D', 'A'): 0,
                               ('D', 'B'): 0,
                               ('D', 'C'): 1,
                               ('D', 'D'): 0,
                               ('D', 'E'): 0,
                               ('D', 'F'): 0,

                               ('E', 'A'): 0,
                               ('E', 'B'): 0,
                               ('E', 'C'): 0,
                               ('E', 'D'): 0,
                               ('E', 'E'): 0,
                               ('E', 'F'): 1,

                               ('F', 'A'): 0,
                               ('F', 'B'): 0,
                               ('F', 'C'): 0,
                               ('F', 'D'): 0,
                               ('F', 'E'): 0,
                               ('F', 'F'): 0,
                               }

    assert actual_confusion_dict == expected_confusion_dict

    normalized_expected_confusion_dict = {('A', 'A'): 2 / 3,
                                          ('A', 'B'): 1 / 3,
                                          ('A', 'C'): 0 / 3,
                                          ('A', 'D'): 0 / 3,
                                          ('A', 'E'): 0 / 3,
                                          ('A', 'F'): 0 / 3,

                                          ('B', 'A'): 0 / 2,
                                          ('B', 'B'): 1 / 2,
                                          ('B', 'C'): 0 / 2,
                                          ('B', 'D'): 1 / 2,
                                          ('B', 'E'): 0 / 2,
                                          ('B', 'F'): 0 / 2,

                                          ('C', 'A'): 0 / 2,
                                          ('C', 'B'): 0 / 2,
                                          ('C', 'C'): 2 / 2,
                                          ('C', 'D'): 0 / 2,
                                          ('C', 'E'): 0 / 2,
                                          ('C', 'F'): 0 / 2,

                                          ('D', 'A'): 0 / 1,
                                          ('D', 'B'): 0 / 1,
                                          ('D', 'C'): 1 / 1,
                                          ('D', 'D'): 0 / 1,
                                          ('D', 'E'): 0 / 1,
                                          ('D', 'F'): 0 / 1,

                                          ('E', 'A'): 0 / 1,
                                          ('E', 'B'): 0 / 1,
                                          ('E', 'C'): 0 / 1,
                                          ('E', 'D'): 0 / 1,
                                          ('E', 'E'): 0 / 1,
                                          ('E', 'F'): 1 / 1,

                                          ('F', 'A'): 0,
                                          ('F', 'B'): 0,
                                          ('F', 'C'): 0,
                                          ('F', 'D'): 0,
                                          ('F', 'E'): 0,
                                          ('F', 'F'): 0,
                                          }

    confusion_matrix_dict, normalized_actual_confusion_dict = confusion_matrix(true=TRUE_SERIES,
                                                                               predicted=PREDICTED_SERIES,
                                                                               normalize=True)

    for key, value in normalized_actual_confusion_dict.items():
        assert abs(normalized_expected_confusion_dict[key] - value) < 0.00001


TRUE_PROB_SERIES = np.array(['F', 'F', 'F', 'T', 'F', 'T', 'T', 'F', 'T', 'F'])
PREDICTED_PROB_SERIES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

THRESHOLDS = [0.45, 0.65]


def test_accuracy_rate_by_prob():
    threshold = 0.45
    actual_rate = accuracy_rate_by_prob(true=TRUE_PROB_SERIES,
                                        predicted_prob=PREDICTED_PROB_SERIES,
                                        threshold=threshold,
                                        pos_label='T')

    expected_rate = 0.6

    assert abs(expected_rate - actual_rate) < 0.0001


def test_recall_rate_by_prob():
    threshold = 0.45
    actual_rate = recall_rate_by_prob(true=TRUE_PROB_SERIES,
                                      predicted_prob=PREDICTED_PROB_SERIES,
                                      threshold=threshold,
                                      pos_label='T')

    expected_rate = 3 / 4

    assert abs(expected_rate - actual_rate) < 0.0001


def test_precision_rate_by_prob():
    threshold = 0.45
    actual_rate = precision_rate_by_prob(true=TRUE_PROB_SERIES,
                                         predicted_prob=PREDICTED_PROB_SERIES,
                                         threshold=threshold,
                                         pos_label='T')

    expected_rate = 3 / 6

    assert abs(expected_rate - actual_rate) < 0.0001


def test_confusion_matrix_by_prob():
    """
    only test for the calculations of the metrics;
    :return:
    """
    actual_confusion_matrix_by_prob = confusion_matrix_by_prob(true=TRUE_PROB_SERIES,
                                                               predicted_prob=PREDICTED_PROB_SERIES,
                                                               thresholds=THRESHOLDS,
                                                               pos_label='T',
                                                               table=False, output_metrics=None)

    actual_results_45 = {'TP': 3, 'FN': 1, 'FP': 3, 'TN': 3}

    actual_results_45['Recall'] = actual_results_45['TP'] / (actual_results_45['TP'] + actual_results_45['FN'])
    actual_results_45['FNR'] = actual_results_45['FN'] / (actual_results_45['TP'] + actual_results_45['FN'])
    actual_results_45['FPR'] = actual_results_45['FP'] / (actual_results_45['FP'] + actual_results_45['TN'])
    actual_results_45['TNR'] = actual_results_45['TN'] / (actual_results_45['FP'] + actual_results_45['TN'])

    actual_results_45['Precision'] = actual_results_45['TP'] / (actual_results_45['TP'] + actual_results_45['FP'])
    actual_results_45['FDR'] = actual_results_45['FP'] / (actual_results_45['TP'] + actual_results_45['FP'])
    actual_results_45['FOR'] = actual_results_45['FN'] / (actual_results_45['FN'] + actual_results_45['TN'])
    actual_results_45['NPV'] = actual_results_45['TN'] / (actual_results_45['FN'] + actual_results_45['TN'])

    actual_results_45['Prevalence'] = 4 / 10
    actual_results_45['Accuracy'] = (actual_results_45['TP'] + actual_results_45['TN']) / (actual_results_45['TP'] +
                                                                                           actual_results_45['FN'] +
                                                                                           actual_results_45['FP'] +
                                                                                           actual_results_45['TN'])

    actual_results_45['LR+'] = actual_results_45['Recall'] / actual_results_45['FPR']
    actual_results_45['LR-'] = actual_results_45['FNR'] / actual_results_45['TNR']

    actual_results_45['DOR'] = actual_results_45['LR+'] / actual_results_45['LR-']
    actual_results_45['F1'] = 2 * (actual_results_45['Recall'] * actual_results_45['Precision']) / (actual_results_45['Recall'] +
                                                                                                    actual_results_45['Precision'])

    for metric_name, metric_value in actual_confusion_matrix_by_prob[0.45].items():
        assert abs(metric_value - actual_results_45[metric_name]) < 0.0001

    actual_results_65 = {'TP': 2, 'FN': 2, 'FP': 2, 'TN': 4}

    actual_results_65['Recall'] = actual_results_65['TP'] / (actual_results_65['TP'] + actual_results_65['FN'])
    actual_results_65['FNR'] = actual_results_65['FN'] / (actual_results_65['TP'] + actual_results_65['FN'])
    actual_results_65['FPR'] = actual_results_65['FP'] / (actual_results_65['FP'] + actual_results_65['TN'])
    actual_results_65['TNR'] = actual_results_65['TN'] / (actual_results_65['FP'] + actual_results_65['TN'])

    actual_results_65['Precision'] = actual_results_65['TP'] / (actual_results_65['TP'] + actual_results_65['FP'])
    actual_results_65['FDR'] = actual_results_65['FP'] / (actual_results_65['TP'] + actual_results_65['FP'])
    actual_results_65['FOR'] = actual_results_65['FN'] / (actual_results_65['FN'] + actual_results_65['TN'])
    actual_results_65['NPV'] = actual_results_65['TN'] / (actual_results_65['FN'] + actual_results_65['TN'])

    actual_results_65['Prevalence'] = 4 / 10
    actual_results_65['Accuracy'] = (actual_results_65['TP'] + actual_results_65['TN']) / (actual_results_65['TP'] +
                                                                                           actual_results_65['FN'] +
                                                                                           actual_results_65['FP'] +
                                                                                           actual_results_65['TN'])

    actual_results_65['LR+'] = actual_results_65['Recall'] / actual_results_65['FPR']
    actual_results_65['LR-'] = actual_results_65['FNR'] / actual_results_65['TNR']

    actual_results_65['DOR'] = actual_results_65['LR+'] / actual_results_65['LR-']
    actual_results_65['F1'] = 2 * (actual_results_65['Recall'] * actual_results_65['Precision']) / (
                actual_results_65['Recall'] +
                actual_results_65['Precision'])

    for metric_name, metric_value in actual_confusion_matrix_by_prob[0.65].items():
        assert abs(metric_value - actual_results_65[metric_name]) < 0.0001


def test_roc():
    """
    make sure the results are consistent with the results of confusion matrix by prob function;
    :return:
    """
    fpr, tpr, thresholds = roc(true=TRUE_PROB_SERIES,
                               predicted_prob=PREDICTED_PROB_SERIES,
                               pos_label='T')

    expected_confusion_matrix_by_prob = confusion_matrix_by_prob(true=TRUE_PROB_SERIES,
                                                                 predicted_prob=PREDICTED_PROB_SERIES,
                                                                 thresholds=thresholds,
                                                                 pos_label='T',
                                                                 table=False, output_metrics=['FPR', 'Recall'])

    for index, threshold in enumerate(thresholds):
        assert abs(fpr[index] - expected_confusion_matrix_by_prob[threshold]['FPR']) < 0.0001
        assert abs(tpr[index] - expected_confusion_matrix_by_prob[threshold]['Recall']) < 0.0001
