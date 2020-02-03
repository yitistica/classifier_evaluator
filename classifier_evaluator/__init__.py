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


def confusion_matrix_by_prob_decorator(func):
    """
    :param func: function, confusion matrix function;
    :return:
    """

    def number_of_decimal(number: float):
        """
        find out
        :param number:
        :return:
        """
        if number > 0.00000001:
            number = 0

        digit_as_str = str(number)

        if '.' not in digit_as_str:
            decimal_no = 0
        else:
            decimal_no = len(digit_as_str.split('.')[1])

        return decimal_no

    def wrapper(*args, **kwargs):
        Confusion_Matrix_full = func(*args, **kwargs)

        MaxNo_threshold_decimal = 0
        for i in Confusion_Matrix_full['thresholds']:
            if number_of_decimal(i) > MaxNo_threshold_decimal:
                MaxNo_threshold_decimal = number_of_decimal(i)

        format_map = {'thresholds': '{:,.' + str(MaxNo_threshold_decimal) + 'f}',
                      'TP': '{:,.0f}', 'FN': '{:,.0f}', 'FP': '{:,.0f}', 'TN': '{:,.0f}',
                      'Recall': '{:.3f}', 'FNR': '{:.3f}', 'FPR': '{:.3f}', 'TNR': '{:.3f}',
                      'Precision': '{:.3f}', 'FOR': '{:.3f}', 'FDR': '{:.3f}', 'NPV': '{:.3f}',
                      'Prevalence': '{:.3f}', 'Accuracy': '{:.3f}',
                      'LR+': '{:.3f}', 'LR-': '{:.3f}', 'DOR': '{:.2f}', 'F1': '{:.2f}'}

        for key, value in format_map.items():
            try:
                Confusion_Matrix_full[key] = Confusion_Matrix_full[key].apply(value.format)
            except:
                pass
        return (Confusion_Matrix_full)

    return (wrapper)