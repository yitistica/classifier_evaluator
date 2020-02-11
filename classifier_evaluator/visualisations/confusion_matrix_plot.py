# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from typing import Optional, Union
from classifier_evaluator.metrics import confusion_matrix
from classifier_evaluator.pre_process import data_type_converter

# rc('text', usetex=True)  # use latext
# Setting chinese encoding:
# plt.rcParams['font.sans-serif']=['Microsoft YaHei']  # plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

_DEFAULT_FIGURE_SIZE = (10, 10)
_DEFAULT_COLORMAP = {'diagonal': 'Greens', 'non_diagonal': 'Oranges', 'summary': 'Blues'}


def plot_confusion_matrix_by_dict(confusion_matrix_dict: dict,
                                  class_order: Optional[list] = None,
                                  **kwargs):
    """
    plot a confusion matrix heat map;
    :param confusion_matrix_dict: dict, a confusion matrix dict, e.g.,
        {('A', 'A'): 0, ('A', 'B'): 1, ...}, the key being ('true class', 'predicted class'), and the value being
        the count of the instances with the true class and predicted class;
    :param class_order: [list, None], if specify, confusion matrix's axes will follow the order given;
    :param kwargs:
        :param colormap: dict, that specify the color map for diagonal, non-diagonal, and summary elements in the matrix
            plot;
        :param figure_size: tuple, e.g., (10, 10), the size of the figure;
        :param subtitle: str, sub title given to the plot;
        :param save: [bool, str], if it is saved to an external .png file;
    :return: plt.show or saved image;
    """

    if 'colormap' in kwargs:
        colormap = kwargs['colormap']
    else:
        colormap = _DEFAULT_COLORMAP

    if 'figure_size' in kwargs:
        figure_size = kwargs['figure_size']
    else:
        figure_size = _DEFAULT_FIGURE_SIZE

    if not class_order:
        class_order = list(set([key[0] for key in confusion_matrix_dict.keys()]))

    # unscaled confusion matrix:
    confusion_matrix = np.ndarray(shape=[len(class_order), len(class_order)], dtype=float)
    for index_true, true_class in enumerate(class_order):
        for index_predicted, predicted_class in enumerate(class_order):
            confusion_matrix[index_true, index_predicted] = \
                confusion_matrix_dict[(true_class, predicted_class)]

    confusion_matrix = np.vstack([confusion_matrix, confusion_matrix.sum(axis=0)])
    confusion_matrix = np.hstack([confusion_matrix, confusion_matrix.sum(axis=1).reshape(7, 1)])

    # confusion matrix scaled by true label:
    confusion_matrix_nor_true = np.divide(confusion_matrix, confusion_matrix[:, -1][:, None],
                                          out=np.zeros_like(confusion_matrix),
                                          where=confusion_matrix[:, -1][:, None] != 0)
    # adjust the last row to be fraction of total population;
    confusion_matrix_nor_true[:, -1] = np.divide(confusion_matrix[:, -1], confusion_matrix[-1, -1],
                                                 out=np.zeros_like(confusion_matrix[:, -1]),
                                                 where=confusion_matrix[-1, -1] != 0)

    # confusion matrix scaled by predicted label:
    confusion_matrix_nor_predicted = np.divide(confusion_matrix, confusion_matrix[-1, :][None, :],
                                               out=np.zeros_like(confusion_matrix),
                                               where=confusion_matrix[-1, :][None, :] != 0)
    # adjust the last row to be fraction of total population;
    confusion_matrix_nor_predicted[-1, :] = np.divide(confusion_matrix[-1, :], confusion_matrix[-1, -1],
                                                      out=np.zeros_like(confusion_matrix[-1, :]),
                                                      where=confusion_matrix[-1, -1] != 0)

    # col names:
    col_names = class_order + ['sum']

    plt.figure(figsize=figure_size)
    cmap_base = plt.get_cmap(colormap['non_diagonal'])
    rgba_base = cmap_base(confusion_matrix_nor_true)

    # replace diagonal in the confusion matrix to different color spectrum;
    cmap_diagonal = plt.get_cmap(colormap['diagonal'])
    rbga_diagonal = cmap_diagonal(np.diag(confusion_matrix_nor_true))
    for diagonal_index, col in enumerate(rbga_diagonal):
        rgba_base[diagonal_index, diagonal_index] = col

    # replace the true column to different color spectrum:
    cmap_sum_true = plt.get_cmap(colormap['summary'])
    rbga_sum_true = cmap_sum_true(confusion_matrix_nor_true[:, -1])
    for index, col in enumerate(rbga_sum_true):
        rgba_base[index, -1] = col

    # replace the true column to different color spectrum:
    cmap_sum_predicted = plt.get_cmap(colormap['summary'])
    rbga_sum_predicted = cmap_sum_predicted(confusion_matrix_nor_predicted[-1, :])
    for index, col in enumerate(rbga_sum_predicted):
        rgba_base[-1, index] = col

    plt.imshow(rgba_base, interpolation='nearest')
    tick_marks = np.arange(len(col_names))
    plt.xticks(tick_marks, col_names)
    plt.yticks(tick_marks, col_names)
    plt.ylim(len(col_names)-0.5, -0.5)  # adjusted for the bug before Matplotlib 3.1.2.

    # add annotation text:
    thresh = confusion_matrix_nor_true.max() / 2
    for i, j in itertools.product(range(confusion_matrix.shape[0] - 1),  # left out the last element;
                                  range(confusion_matrix.shape[1] - 1)):

        if i == j:  # annotation for instance with true == predicted;
            num_annotation = r"$\bf{" + str(int(confusion_matrix[i, j])) + "}$" + "\n"

            recall_num = str(round(confusion_matrix_nor_true[i, j], 3))
            recall_annotation = recall_num + r"($\mathit{Recall}$)" + '\n'

            precision_num = str(round(confusion_matrix_nor_predicted[i, j], 3))
            precision_annotation = precision_num + r"($\mathit{Precision}$)"

            annotation_text = num_annotation + recall_annotation + precision_annotation

            plt.text(j, i, annotation_text,
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")

        else:
            num_annotation = r"$\bf{" + str(int(confusion_matrix[i, j])) + "}$" + "\n"

            recall_num = str(round(confusion_matrix_nor_true[i, j], 3))
            recall_annotation = recall_num + r"($\mathit{/True}$)" + '\n'

            precision_num = str(round(confusion_matrix_nor_predicted[i, j], 3))
            precision_annotation = precision_num + r"($\mathit{/Predicted}$)"

            annotation_text = num_annotation + recall_annotation + precision_annotation

            plt.text(j, i, annotation_text,
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")

    # add text for the last col (sum):
    last_index = confusion_matrix.shape[0] - 1
    for i in range(confusion_matrix.shape[0]):
        num_annotation = r"$\bf{" + str(int(confusion_matrix[i, -1])) + "}$" + "\n"
        percentage_annotation = r"$\bf{" + str(round(confusion_matrix_nor_true[i, -1] * 100, 2)) + r"\%}$"

        annotation_text = num_annotation + percentage_annotation
        plt.text(last_index, i, annotation_text,
                 horizontalalignment="center",
                 verticalalignment="center",

                 color="white" if confusion_matrix_nor_true[i, -1] > thresh else "black")

    for j in range(confusion_matrix.shape[1]):
        num_annotation = r"$\bf{" + str(int(confusion_matrix[-1, j])) + "}$" + "\n"
        percentage_annotation = r"$\bf{" + str(round(confusion_matrix_nor_predicted[-1, j] * 100, 2)) + r"\%}$"

        annotation_text = num_annotation + percentage_annotation
        plt.text(j, last_index, annotation_text,
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if confusion_matrix_nor_predicted[-1, j] > thresh else "black")

    # plt.tight_layout()
    if 'subtitle' in kwargs:
        title_text = f"Confusion Matrix Plot ({kwargs['subtitle']})"
        saved_file_name = f"confusion_matrix_plot_{kwargs['subtitle']}.png"
    else:
        title_text = f"Confusion Matrix Plot"
        saved_file_name = f"confusion_matrix_plot.png"

    plt.title(title_text)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if 'save' in kwargs:
        save = kwargs['save']
        if isinstance(save, str):
            plt.savefig(save)
        else:
            plt.savefig(saved_file_name)

    plt.show()


def plot_confusion_matrix_on_predicted_series(true_series: Union[pd.Series, np.ndarray, list],
                                              predicted_series: Union[pd.Series, np.ndarray, list],
                                              class_order: Optional[list] = None,
                                              **kwargs):
    """
    run the above confusion plot function after running confusion matrix function;
    :param true_series: [pd.Series, np.ndarray, list, None], a series of true classes;
    :param predicted_series: [pd.Series, np.ndarray, list, None], a series of predicted classes;
    :param class_order: refer class_order param above;
    :param kwargs: refer kwargs above;
    :return: plt.show or saved image;
    """
    true_series = data_type_converter(series=true_series)
    predicted_series = data_type_converter(series=predicted_series)
    confusion_matrix_dict = confusion_matrix(true=true_series,
                                             predicted=predicted_series, normalize=False)

    plot_confusion_matrix_by_dict(confusion_matrix_dict=confusion_matrix_dict,
                                  class_order=class_order, **kwargs)


TRUE_SERIES = np.array(['A', 'A', 'A', 'B', 'B', 'C', 'C', 'D', 'E'])
PREDICTED_SERIES = np.array(['A', 'A', 'B', 'B', 'D', 'C', 'C', 'C', 'F'])


plot_confusion_matrix_on_predicted_series(TRUE_SERIES, PREDICTED_SERIES, class_order=['A', 'B', 'C', 'D', 'E', 'F'], save='abc.png')