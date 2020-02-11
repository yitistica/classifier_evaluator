import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from classifier_evaluator.metrics import roc, roc_auc
from classifier_evaluator.pre_process import data_type_converter
# setting standard plot size:
_DEFAULT_FIGURE_SIZE = (10, 10)
_DEFAULT_COLOR_PALETTE = "jet"


def plot_roc(roc_dict: dict, **kwargs):
    """
    plot roc curves;
    :param roc_dict: dict, that contains series needed for the plot; i.e.,
        set 1: given fpr and tpr:
            {model_name: {"fpr": [pd.Series, np.ndarray, list], "tpr": [pd.Series, np.ndarray, list], ...}}
        set 2: given true series and predicted_prob series:
            {model_name: {"true": [pd.Series, np.ndarray, list], "predicted_prob": [pd.Series, np.ndarray, list], ...}}
    :param kwargs:
        :param figure_size: tuple, e.g., (10, 10), the size of the figure;
        :param subtitle: str, sub title given to the plot;
        :param save: [bool, str], if it is saved to an external .png file;
    :return: plt.show or saved image;
    """
    roc_element_dict = dict()
    for model_name, series_dict in roc_dict.items():
        if ('fpr' in series_dict) and ('tpr' in series_dict):
            roc_element_dict[model_name] = {'fpr': data_type_converter(series_dict['fpr']),
                                            'tpr': data_type_converter(series_dict['tpr'])}
        elif ('true' in series_dict) and ('predicted_prob' in series_dict):
            if 'pos_label' in series_dict:
                fpr, tpr, thresholds = roc(true=data_type_converter(series_dict['true']),
                                           predicted_prob=data_type_converter(series_dict['predicted_prob']),
                                           pos_label=series_dict['pos_label'])
            else:
                fpr, tpr, thresholds = roc(true=data_type_converter(series_dict['true']),
                                           predicted_prob=data_type_converter(series_dict['predicted_prob']))

            roc_element_dict[model_name] = {'fpr': fpr,
                                            'tpr': tpr}
        else:
            raise KeyError(f"series dict does not contain needed series ({list(series_dict.keys())})")

    # compute auc if not present:
    for model_name, series_dict in roc_element_dict.items():
        if 'auc' in roc_dict[model_name]:
            series_dict['auc'] = roc_dict[model_name]['auc']
        else:
            series_dict['auc'] = roc_auc(**series_dict)

    # plot set up:
    if 'figure_size' in kwargs:
        figure_size = kwargs['figure_size']
    else:
        figure_size = _DEFAULT_FIGURE_SIZE

    fig, ax = plt.subplots(figsize=figure_size)
    n_model = len(roc_element_dict)
    cm_subsection = np.linspace(0, 1, n_model)
    cmap = cm.get_cmap(_DEFAULT_COLOR_PALETTE)
    colors = [cmap(x) for x in cm_subsection]

    # setting label size:
    axis_label_size = int(min(figure_size) * 1.6)
    title_label_size = int(min(figure_size) * 2.1)
    line_width = int(min(figure_size) / 6)

    ax.set_xlabel('False Positive Rate', fontsize= axis_label_size)
    ax.set_ylabel('True Positive Rate', fontsize= axis_label_size)

    if 'subtitle' in kwargs:
        title_text = f"Reciver Operating Characteristic Plot ({kwargs['subtitle']})"
        saved_file_name = f"roc_{kwargs['subtitle']}.png"
    else:
        title_text = f"Reciver Operating Characteristic Plot"
        saved_file_name = f"roc.png"

    ax.title.set_text(title_text)
    ax.title.set_fontsize(title_label_size)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    ax.grid(True)

    _index = -1
    for model_name, series_dict in roc_element_dict.items():
        _index += 1
        ax.plot(series_dict['fpr'], series_dict['tpr'],
                lw=line_width, alpha=1,
                label=f"Model {model_name} (AUC = {str(round(series_dict['auc'], 2))})",
                color=colors[_index])

    ax.plot([0, 1], [0, 1], linestyle='--', lw=line_width, color='navy', label='Luck', alpha=.8)
    ax.legend(loc="lower right")

    if 'save' in kwargs:
        save = kwargs['save']
        if isinstance(save, str):
            plt.savefig(save)
        else:
            plt.savefig(saved_file_name)

    plt.show()


TRUE_PROB_SERIES = np.array(['F', 'F', 'F', 'T', 'F', 'T', 'T', 'F', 'T', 'F'])
PREDICTED_PROB_SERIES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
PREDICTED_PROB_SERIES_2 = np.array([0, 0.1, 0.15, 0.7, 0.5, 0.4, 0.6, 0.8, 0.7, 1])

plot_roc({'a': {'true': TRUE_PROB_SERIES, 'predicted_prob': PREDICTED_PROB_SERIES, 'pos_label': 'T'},
          'b': {'true': TRUE_PROB_SERIES, 'predicted_prob': PREDICTED_PROB_SERIES_2, 'pos_label': 'T'}})