import numpy as np
import matplotlib.pyplot as plt
import itertools
from typing import Union, Tuple, Optional
from classifier_evaluator.metrics import normalize_confusion_matrix

# Setting chinese encoding:
# plt.rcParams['font.sans-serif']=['Microsoft YaHei']  # plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

_DEFAULT_FIGURE_SIZE = (10, 10)
_DEFAULT_COLORMAP = plt.get_cmap('Blues')
_DEFAULT_COLOR_PALETTE = "husl"


def plot_normalized_confusion_matrix_by_dict(normalized_confusion_matrix_dict: dict,
                                             class_order: Optional[list] = None):

    normalized_confusion_matrix_dict

    if not class_order:
        class_order = 

    plt.figure(figsize=_DEFAULT_FIGURE_SIZE)
    plt.imshow(normalised_confusion_matrix, interpolation='nearest', cmap=_DEFAULT_COLORMAP)
    plt.title('Confusion Matrix Plot')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # add annotation text:
    thresh = normalised_confusion_matrix.max() / 2.
    for i, j in itertools.product(range(normalised_confusion_matrix.shape[0]),
                                  range(normalised_confusion_matrix.shape[1])):
        plt.text(j, i, '%0.3f with %d' % (normalised_confusion_matrix[i, j], unnormalised_confusion_matrix[i, j]),
                 horizontalalignment="center",
                 color="white" if normalised_confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()