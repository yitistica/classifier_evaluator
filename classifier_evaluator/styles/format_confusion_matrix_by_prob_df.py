"""
reformat confusion matrix by prob data frame;
"""
import pandas as pd
from typing import Optional


_CONFUSION_MATRIX_DIGIT_FORMAT = {'TP': '{:,.0f}',
                                  'FN': '{:,.0f}',
                                  'FP': '{:,.0f}',
                                  'TN': '{:,.0f}',
                                  'Recall': '{:.3f}',
                                  'FNR': '{:.3f}',
                                  'FPR': '{:.3f}',
                                  'TNR': '{:.3f}',
                                  'Precision': '{:.3f}',
                                  'FOR': '{:.3f}',
                                  'FDR': '{:.3f}',
                                  'NPV': '{:.3f}',
                                  'Prevalence': '{:.3f}',
                                  'Accuracy': '{:.3f}',
                                  'LR+': '{:.3f}',
                                  'LR-': '{:.3f}',
                                  'DOR': '{:.2f}',
                                  'F1': '{:.2f}'}


_DEFAULT_METRIC_ORDER = ['TP', 'FN', 'FP', 'TN',
                         'Recall', 'FNR', 'FPR', 'TNR',
                         'Precision', 'FOR', 'FDR', 'NPV',
                         'Prevalence', 'Accuracy', 'LR+', 'LR-', 'DOR', 'F1']


def _reformat_digit_confusion_matrix_by_prob(metrics_by_thresholds_df: pd.DataFrame,
                                             digit_format: Optional[dict] = None) -> pd.DataFrame:
    """
    reformat the digit representations confusion matrix by prob data frame by:
        1. finding maximum number of decimal places for threshold series, and set such precision to each threshold,
        2. for format in max_digit_format, apply the setting to each corresponding series;

    :param metrics_by_thresholds_df: pandas.DataFrame, a confusion matrix metric data frame;
    :param digit_format: dict, format map that gives the precision/decimal places of the series;
    :return: reformatted metric data frame;
    """
    def get_number_of_decimal_places(number: float) -> int:
        """
        find out how many number of decimal places of a given number;
        :param number: float, a number;
        :return: number of decimal places of the number;
        """
        if number < 0.000001:
            number = 0

        digit_as_str = str(number)

        if '.' not in digit_as_str:
            decimal_no = 0
        else:
            decimal_no = len(digit_as_str.split('.')[1])

        return decimal_no

    max_no_threshold_decimal = max(metrics_by_thresholds_df['threshold'].apply(get_number_of_decimal_places))

    threshold_decimal_format = '{:,.' + str(max_no_threshold_decimal) + 'f}'
    metrics_by_thresholds_df['threshold'] = metrics_by_thresholds_df['threshold'].apply(threshold_decimal_format.format)

    if not digit_format:
        digit_format = _CONFUSION_MATRIX_DIGIT_FORMAT

    for metric, decimal_format in digit_format.items():
        if metric in metrics_by_thresholds_df.columns:
            metrics_by_thresholds_df[metric] = metrics_by_thresholds_df[metric].apply(decimal_format.format)

    return metrics_by_thresholds_df


def _convert_confusion_matrix_by_prob_to_table(metrics_by_thresholds: dict,
                                               metric_order: Optional[list] = None) -> pd.DataFrame:
    """
    reformat the confusion matrix by prob from dict format to data frame format;

    :param metrics_by_thresholds: a (confusion matrix) metric dict (with other additional metrics) by different thresholds;
    :param metric_order: list, the order by which metrics are presented after threshold series;
        if the item is not in metric order, but presented in metrics_by_thresholds, it will be placed after the last item
            of metric order with random placement;
        if the item is in the metric order but not presented in metrics_by_thresholds, it will be ignored;
    :return: confusion_matrix_by_prob_to_table, pandas.DataFrame;
    """
    for threshold, metrics in metrics_by_thresholds.items():
        metrics['threshold'] = threshold

    if not metric_order:
        metric_order = _DEFAULT_METRIC_ORDER

    metric_order = ['threshold'] + [column for column in metric_order if column in list(metrics_by_thresholds.values())[0]]

    metric_order += [column for column in list(metrics_by_thresholds.values())[0] if column not in metric_order]

    metrics_by_thresholds = list(metrics_by_thresholds.values())
    metrics_by_thresholds_df = pd.DataFrame(metrics_by_thresholds)

    metrics_by_thresholds_df = metrics_by_thresholds_df[metric_order]

    metrics_by_thresholds_df = metrics_by_thresholds_df.sort_values(by='threshold', ascending=True)
    metrics_by_thresholds_df = metrics_by_thresholds_df.reset_index(drop=True)

    return metrics_by_thresholds_df


def convert_confusion_matrix_by_prob_to_table_with_reformat_precision(metrics_by_thresholds: dict,
                                                                      metric_order: Optional[list] = None,
                                                                      digit_format: Optional[dict] = None) -> pd.DataFrame:
    """
    wrap function that perform conversion of confusion matrix by prob from dict to pandas.DataFrame with reformatting;

    :param metrics_by_thresholds: dict, read above;
    :param metric_order: list, read above;
    :param digit_format: dict, read above;
    :return: confusion_matrix_by_prob_to_table, pandas.DataFrame;
    """

    metrics_by_thresholds_df = _convert_confusion_matrix_by_prob_to_table(metrics_by_thresholds=metrics_by_thresholds,
                                                                          metric_order=metric_order)

    metrics_by_thresholds_df = _reformat_digit_confusion_matrix_by_prob(metrics_by_thresholds_df=metrics_by_thresholds_df,
                                                                        digit_format=digit_format)

    return metrics_by_thresholds_df


