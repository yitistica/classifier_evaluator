from typing import Union, Optional
import numpy as np
import pandas as pd

from classifier_evaluator.pre_process import data_type_converter
from classifier_evaluator.metrics import accuracy_rate_by_prob, recall_rate_by_prob, precision_rate_by_prob, \
    confusion_matrix_by_prob, roc, roc_auc


_DEFAULT_THRESHOLD = 0.5

_DEFAULT_THRESHOLDS = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]

_TRUE_SERIES_NAME = 'true'


class OccupiedSeriesNameError(Exception):
    def __init__(self, message):
        super().__init__(message)


class SeriesNotFound(Exception):
    def __init__(self, series_name):
        super().__init__(f"series {series_name} not found in panel.")


class ClassifierDataPanel(object):
    """
    data panel that manages a true series and predicted prob series;
    """

    def __init__(self,
                 true_series: Optional[Union[pd.Series, np.ndarray, list]] = None,
                 pos_label: Union[bool, str, int] = True,
                 *predicted_prob_series: Union[pd.Series, np.ndarray, list],
                 **named_predicted_prob_series: Union[pd.Series, np.ndarray, list]):
        """
        :param true_series: [pd.Series, np.ndarray, list, None], a series of true classes;
        :param pos_label: [str, bool, int], positive class label, label that is considered as the positive class;
        :param predicted_prob_series: [pd.Series, np.ndarray, list, None],
            a series of predicted probabilities of being the positive class (without a given name);
        :param named_predicted_prob_series: [pd.Series, np.ndarray, list, None],
            a series of predicted probabilities of being the positive class (with a given name);
        """

        self.series = dict()
        self.pos_label = None

        if true_series:
            self.inject_true_series(series=true_series, pos_label=pos_label)

        for series in predicted_prob_series:
            series_name = self._check_series_name(proposed_name=None)
            self.series[series_name] = series

        for series_name, series in named_predicted_prob_series.items():
            series_name = self._check_series_name(proposed_name=series_name)
            self.series[series_name] = series

    def _check_series_name(self, proposed_name: Union[str, None]):
        """
        check and proposed name for a series into the panel is valid;
        :param proposed_name: [str, None], if a proposed name for the series;
        :return: a generated name or the valid proposed name;
        """
        prefix = 'predicted_prob'
        if not proposed_name:
            suffix_index = 1
            while f"{prefix}_{suffix_index}" in self.series:
                suffix_index += 1
            proposed_name = f"{prefix}_{suffix_index}"

        elif (proposed_name == _TRUE_SERIES_NAME) or (proposed_name in self.series):
            raise OccupiedSeriesNameError(f"Series {proposed_name} is already occupied, use other name.")

        else:
            proposed_name = proposed_name

        return proposed_name

    def rename_series(self, new_names: dict):
        """
        re-naming series;
        :param new_names: dict, {previous name: new name};
        :return:
        """
        for previous_name, new_name in new_names.items():
            if previous_name in self.series:
                new_name = self._check_series_name(proposed_name=new_name)
                self.series[new_name] = self.series.pop(previous_name)

    def inject_true_series(self,
                           series: Union[pd.Series, np.ndarray, list],
                           pos_label: Union[bool, str, int] = True):
        """
        inject a true series into the panel;
        :param series: [pd.Series, np.ndarray, list, None], a series of true classes;
        :param pos_label: [str, bool, int], positive class label, label that is considered as the positive class;
        :return:
        """
        self.series[_TRUE_SERIES_NAME] = data_type_converter(series=series)
        self.pos_label = pos_label

    def inject_predicted_prob(self,
                              series: Union[pd.Series, np.ndarray, list],
                              series_name: Optional[str] = None):
        """
        inject a predicted prob series into the panel;
        :param series: [pd.Series, np.ndarray, list, None], a series of true classes;
        :param series_name: [str, None], the name given to this series, if not given, auto generated name is given;
        :return:
        """

        series_name = self._check_series_name(proposed_name=series_name)
        self.series[series_name] = data_type_converter(series=series)

    def __add__(self, series: Union[pd.Series, np.ndarray, list]):
        """
        add a series into the panel using an operator;
        :param series: [pd.Series, np.ndarray, list, None], a series of true classes;
        :return: the panel instance;
        """
        if not self.series:
            self.inject_true_series(series=series)
        else:
            self.inject_predicted_prob(series=series,
                                       series_name=None)

        return self

    def remove_series(self, series_name: str):
        """
        remove a series from the panel;
        :param series_name: str, name of the series to be removed;
        :return:
        """
        if series_name not in self.series:
            raise SeriesNotFound(series_name)

        del self.series[series_name]


class ClassifierEvalPanel(ClassifierDataPanel):
    """
    panel that perform classifier evaluations;
    """

    def __init__(self, thresholds: Optional[Union[pd.Series, np.ndarray, list]] = None,
                 *args, **kwargs):
        """
        params in ClassifierDataPanel;

        """
        super().__init__(*args, **kwargs)
        self.thresholds = None
        self.set_default_thresholds(thresholds=thresholds)

    def set_default_thresholds(self, thresholds: Optional[Union[pd.Series, np.ndarray, list]] = None):
        """
        set default thresholds for confusion matrix by prob;
        :param thresholds: [pd.Series, np.ndarray, list, None], a series of thresholds over which predicted prob will
        be classified as positive, if None is given, it will use the default threshold series;
        :return:
        """
        if not thresholds:
            thresholds = _DEFAULT_THRESHOLDS

        self.thresholds = data_type_converter(series=thresholds)

    def accuracy_rate_by_prob(self,
                              thresholds: Optional[Union[pd.Series, np.ndarray, list]] = None,
                              focus_series: Optional[Union[list, set]] = None):
        """
        compute accuracy rate for different thresholds for different predicted prob series;
        :param thresholds: [pd.Series, np.ndarray, list, None], a series of thresholds over which predicted prob will
        be classified as positive, if None is given, it will use the default threshold series;
        :param focus_series: [list, set], names of the predicted prob series that metrics are run on;
        :return: accuracy rates, dict, {series_name: {threshold: {'Accuracy': float}}, .. };
        """
        if not thresholds:
            thresholds = _DEFAULT_THRESHOLDS

        if not focus_series:
            selected_series = {series_name: series for series_name, series in self.series.items()
                               if series_name != _TRUE_SERIES_NAME}
        else:
            selected_series = {series_name: series for series_name, series in self.series.items()
                               if series_name in focus_series}

        results = dict()
        for series_name, predicted_prob in selected_series.items():
            results[series_name] = dict()
            for threshold in thresholds:
                accuracy_rate = accuracy_rate_by_prob(true=self.series[_TRUE_SERIES_NAME],
                                                      predicted_prob=predicted_prob,
                                                      threshold=threshold,
                                                      pos_label=self.pos_label)

                results[series_name][threshold] = {'Accuracy': accuracy_rate}

        return results

    def recall_by_prob(self,
                       thresholds: Optional[Union[pd.Series, np.ndarray, list]] = None,
                       focus_series: Optional[Union[list, set]] = None):
        """
        compute recall rate for different thresholds for different predicted prob series;
        :param thresholds: [pd.Series, np.ndarray, list, None], a series of thresholds over which predicted prob will
        be classified as positive, if None is given, it will use the default threshold series;
        :param focus_series: [list, set], names of the predicted prob series that metrics are run on;
        :return: recall rates, dict, {series_name: {threshold: {'Recall': float}}, .. };
        """
        if not thresholds:
            thresholds = _DEFAULT_THRESHOLDS

        if not focus_series:
            selected_series = {series_name: series for series_name, series in self.series.items()
                               if series_name != _TRUE_SERIES_NAME}
        else:
            selected_series = {series_name: series for series_name, series in self.series.items()
                               if series_name in focus_series}

        results = dict()
        for series_name, predicted_prob in selected_series.items():
            results[series_name] = dict()
            for threshold in thresholds:
                recall_rate = recall_rate_by_prob(true=self.series[_TRUE_SERIES_NAME],
                                                  predicted_prob=predicted_prob,
                                                  threshold=threshold,
                                                  pos_label=self.pos_label)

                results[series_name][threshold] = {'Recall': recall_rate}

        return results

    def precision_by_prob(self,
                          thresholds: Optional[list] = None,
                          focus_series: Optional[Union[list, set]] = None):
        """
        compute precision rate for different thresholds for different predicted prob series;
        :param thresholds: [pd.Series, np.ndarray, list, None], a series of thresholds over which predicted prob will
        be classified as positive, if None is given, it will use the default threshold series;
        :param focus_series: [list, set], names of the predicted prob series that metrics are run on;
        :return: precision rates, dict, {series_name: {threshold: {'precision': float}}, .. };
        """
        if not thresholds:
            thresholds = _DEFAULT_THRESHOLDS

        if not focus_series:
            selected_series = {series_name: series for series_name, series in self.series.items()
                               if series_name != _TRUE_SERIES_NAME}
        else:
            selected_series = {series_name: series for series_name, series in self.series.items()
                               if series_name in focus_series}

        results = dict()
        for series_name, predicted_prob in selected_series.items():
            results[series_name] = dict()
            for threshold in thresholds:
                recall_rate = precision_rate_by_prob(true=self.series[_TRUE_SERIES_NAME],
                                                     predicted_prob=predicted_prob,
                                                     threshold=threshold,
                                                     pos_label=self.pos_label)

                results[series_name][threshold] = {'Precision': recall_rate}

        return results

    def confusion_matrix_by_prob(self,
                                 thresholds: Optional[list] = None,
                                 output_metrics: Optional[list] = None,
                                 focus_series: Optional[Union[list, set]] = None,
                                 table: bool = True):
        """
        confusion matrix for binary classification according to a given set of thresholds;
        :param thresholds: [pd.Series, np.ndarray, list, None], a series of thresholds over which predicted prob will
        be classified as positive, if None is given, it will use the default threshold series;
        :param output_metrics: [list, None], metrics to be outputted if selected;
        :param focus_series: [list, set], names of the predicted prob series that metrics are run on;
        :param table: bool, if exported as a pd table table;
        :return: confusion matrix results, [dict, pandas.DataFrame];
        """
        if not thresholds:
            thresholds = self.thresholds

        if not focus_series:
            selected_series = {series_name: series for series_name, series in self.series.items()
                               if series_name != _TRUE_SERIES_NAME}
        else:
            selected_series = {series_name: series for series_name, series in self.series.items()
                               if series_name in focus_series}

        results = dict()
        for series_name, predicted_prob in selected_series.items():
            confusion_matrix_by_threshold = confusion_matrix_by_prob(true=self.series[_TRUE_SERIES_NAME],
                                                                     predicted_prob=predicted_prob,
                                                                     thresholds=thresholds,
                                                                     pos_label=self.pos_label,
                                                                     output_metrics=output_metrics,
                                                                     table=table)

            results[series_name] = {'metrics_by_thresholds': confusion_matrix_by_threshold}

        return results

    def roc(self,
            focus_series: Optional[Union[list, set]] = None,
            auc: bool = False):
        """
        compute roc series;
        :param focus_series: [list, set], names of the predicted prob series that metrics are run on;
        :param auc: bool, if auc is also computed;
        :return: roc series, dict; {series_name: {'roc': {'fpr': series, 'tpr': series, 'thresholds': series}}, ...}
        """
        if not focus_series:
            selected_series = {series_name: series for series_name, series in self.series.items()
                               if series_name != _TRUE_SERIES_NAME}
        else:
            selected_series = {series_name: series for series_name, series in self.series.items()
                               if series_name in focus_series}

        results = dict()
        for series_name, predicted_prob in selected_series.items():
            fpr, tpr, thresholds = roc(true=self.series[_TRUE_SERIES_NAME],
                                       predicted_prob=predicted_prob,
                                       pos_label=self.pos_label)

            results[series_name] = {'roc': {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}}

            if auc:
                results[series_name]['auc'] = roc_auc(fpr=fpr, tpr=tpr, thresholds=thresholds)

        return results

