from typing import Union
import numpy as np
import pandas as pd

from classifier_evaluator.metrics import accuracy_rate_by_prob, \
    confusion_matrix_by_prob

_DEFAULT_THRESHOLD = 0.5

_DEFAULT_THRESHOLDS = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]

_TRUE_SERIES_NAME = 'true'


def _data_type_converter(series: Union[pd.Series, np.ndarray, list]) -> np.ndarray:
    if isinstance(series, list):
        series = np.array(series)
    elif isinstance(series, np.ndarray):
        pass
    elif isinstance(series, pd.Series):
        series = series.values()

    return series


class OccupiedSeriesNameError(Exception):
    def __init__(self, message):
        super().__init__(message)


class SeriesNotFound(Exception):
    def __init__(self, series_name):
        super().__init__(f"series {series_name} not found in panel.")


class ClassifierDataPanel(object):
    def __init__(self,
                 true_series=None,
                 pos_label: Union[bool, str, int] = True,
                 *predicted_prob_series: Union[pd.Series, np.ndarray, list],
                 **named_predicted_prob_series: Union[pd.Series, np.ndarray, list]):

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

        self.focus_series = set()

    def _check_series_name(self, proposed_name: Union[str, None]):
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
        for previous_name, new_name in new_names.items():
            if previous_name in self.series:
                new_name = self._check_series_name(proposed_name=new_name)
                self.series[new_name] = self.series.pop(previous_name)

    def inject_true_series(self,
                           series: Union[pd.Series, np.ndarray, list],
                           pos_label: Union[bool, str, int] = True):

        self.series[_TRUE_SERIES_NAME] = _data_type_converter(series=series)
        self.pos_label = pos_label

    def inject_predicted_prob(self,
                              series: Union[pd.Series, np.ndarray, list],
                              series_name: Union[str, None] = None):

        series_name = self._check_series_name(proposed_name=series_name)
        self.series[series_name] = _data_type_converter(series=series)

    def __add__(self, series: Union[pd.Series, np.ndarray, list]):
        if not self.series:
            self.inject_true_series(series=series)
        else:
            self.inject_predicted_prob(series=series,
                                       series_name=None)

        return self

    def remove_series(self, series_name):
        if series_name not in self.series:
            raise SeriesNotFound(series_name)

        del self.series[series_name]
        self.focus_series.discard(series_name)

    def focus_on(self, *series_names):
        self.focus_series = set()
        for series_name in series_names:
            if series_name in self.series:
                self.focus_series.add(series_name)


class ClassifierEvalPanel(ClassifierDataPanel):
    def __init__(self, thresholds: Union[pd.Series, np.ndarray, list, None] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thresholds = None
        self.set_default_thresholds(thresholds=thresholds)

    def set_default_thresholds(self, thresholds: Union[pd.Series, np.ndarray, list, None] = None):
        if not thresholds:
            thresholds = _DEFAULT_THRESHOLDS

        self.thresholds = _data_type_converter(series=thresholds)

    def accuracy_rate_by_prob(self, threshold: Union[float, None] = None):
        results = dict()
        if not threshold:
            threshold = _DEFAULT_THRESHOLD

        for series_name, predicted_prob in self.series.items():
            if series_name != _TRUE_SERIES_NAME:
                accuracy_rate = accuracy_rate_by_prob(true=self.series[_TRUE_SERIES_NAME],
                                                      predicted_prob=predicted_prob,
                                                      threshold=threshold,
                                                      pos_label=self.pos_label)

                results[series_name] = {'Accuracy': accuracy_rate}

        return results

    def confusion_matrix_by_prob(self, thresholds: Union[list, None] = None,
                                 output_metrics: Union[list, None] = None,
                                 table: bool = True):

        results = dict()

        if not thresholds:
            thresholds = self.thresholds

        for series_name, predicted_prob in self.series.items():
            if series_name != _TRUE_SERIES_NAME:
                confusion_matrix_by_threshold = confusion_matrix_by_prob(true=self.series[_TRUE_SERIES_NAME],
                                                                         predicted_prob=predicted_prob,
                                                                         thresholds=thresholds,
                                                                         pos_label=self.pos_label,
                                                                         output_metrics=output_metrics,
                                                                         table=table)

                results[series_name] = {'metrics_by_thresholds': confusion_matrix_by_threshold}

        return results

    def roc(self, thresholds: Union[list, None] = None,
            output_metrics: Union[list, None] = None,
            table: bool = True):

        results = dict()

        if not thresholds:
            thresholds = self.thresholds

        for series_name, predicted_prob in self.series.items():
            if series_name != _TRUE_SERIES_NAME:
                confusion_matrix_by_threshold = confusion_matrix_by_prob(true=self.series[_TRUE_SERIES_NAME],
                                                                         predicted_prob=predicted_prob,
                                                                         thresholds=thresholds,
                                                                         pos_label=self.pos_label,
                                                                         output_metrics=output_metrics,
                                                                         table=table)

                results[series_name] = {'metrics_by_thresholds': confusion_matrix_by_threshold}

        return results