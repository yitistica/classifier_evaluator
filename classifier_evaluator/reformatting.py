import pandas as pd

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


def reformat_digit_confusion_matrix_by_prob(metrics_by_thresholds_df, max_digit_format):
    """

    :param metrics_by_thresholds_df:
    :param max_digit_format:
    :return:
    """
    def number_of_decimal(number: float):
        """
        find out
        :param number:
        :return:
        """
        if number > 0.000001:
            number = 0

        digit_as_str = str(number)

        if '.' not in digit_as_str:
            decimal_no = 0
        else:
            decimal_no = len(digit_as_str.split('.')[1])

        return decimal_no

    max_no_threshold_decimal = max(metrics_by_thresholds_df['thresholds'].apply(number_of_decimal))

    threshold_decimal_format = '{:,.' + str(max_no_threshold_decimal) + 'f}'
    metrics_by_thresholds_df['thresholds'] = metrics_by_thresholds_df['thresholds'].apply(threshold_decimal_format)

    for metric, decimal_format in max_digit_format.items():
        if metric in metrics_by_thresholds_df.columns:
            metrics_by_thresholds_df[metric] = metrics_by_thresholds_df[metric].apply(decimal_format)

    return metrics_by_thresholds_df


def convert_confusion_matrix_by_prob_to_table(metrics_by_thresholds: dict, metric_order):
    """

    :param metrics_by_thresholds:
    :param metric_order:
    :return:
    """
    for threshold, metrics in metrics_by_thresholds.items():
        metrics_by_thresholds['threshold'] = threshold

    metric_order = ['threshold'] + [column for column in metric_order if column in list(metrics_by_thresholds.values())[0]]

    metrics_by_thresholds = list(metrics_by_thresholds.values())
    metrics_by_thresholds_df = pd.DataFrame(metrics_by_thresholds)

    metrics_by_thresholds_df = metrics_by_thresholds_df[metric_order]

    metrics_by_thresholds_df = metrics_by_thresholds_df.sort_values(by='threshold', ascending=True)

    return metrics_by_thresholds_df

