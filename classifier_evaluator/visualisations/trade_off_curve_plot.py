import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib import gridspec
from classifier_evaluator.metrics import roc_margins
from classifier_evaluator.pre_process import data_type_converter
_DEFAULT_FIGURE_SIZE = (10, 10)
_DEFAULT_COLOR_PALETTE = "jet"


def plot_trade_off_curves(roc_dict: dict, **kwargs):
    """
    plot roc margin curve;
    :param roc_dict: dict, that contains series needed for the plot; i.e.,
        set 1: given fpr and tpr:
            {model_name: {"fpr": [pd.Series, np.ndarray, list],
                          "tpr": [pd.Series, np.ndarray, list],
                          "thresholds": [pd.Series, np.ndarray, list], ...}}
        set 2: given true series and predicted_prob series:
            {model_name: {"true": [pd.Series, np.ndarray, list], "predicted_prob": [pd.Series, np.ndarray, list], ..., ...}}
    :param kwargs:
        :param figure_size: tuple, e.g., (10, 10), the size of the figure;
        :param subtitle: str, sub title given to the plot;
        :param save: [bool, str], if it is saved to an external .png file;
    :return: plt.show or saved image;
    """
    roc_element_dict = dict()
    for model_name, series_dict in roc_dict.items():
        if ('fpr' in series_dict) and ('tpr' in series_dict) and ('thresholds' in series_dict):
            roc_element_dict[model_name] = series_dict
        elif ('true' in series_dict) and ('predicted_prob' in series_dict):
            if 'pos_label' in series_dict:
                roc_margin_dict = roc_margins(true=data_type_converter(series_dict['true']),
                                              predicted_prob=data_type_converter(series_dict['predicted_prob']),
                                              pos_label=series_dict['pos_label'])
            else:
                roc_margin_dict = roc_margins(true=data_type_converter(series_dict['true']),
                                              predicted_prob=data_type_converter(series_dict['predicted_prob']))

            roc_element_dict[model_name] = roc_margin_dict
        else:
            raise KeyError(f"series dict does not contain needed series ({list(series_dict.keys())})")

    # plot set up:
    if 'figure_size' in kwargs:
        figure_size = kwargs['figure_size']
    else:
        figure_size = _DEFAULT_FIGURE_SIZE

    fig = plt.figure(figsize=figure_size)
    gs = gridspec.GridSpec(101, 1, figure=fig)  # set height ratio: 2:1
    ax0 = fig.add_subplot(gs[0:65, :])
    ax1 = fig.add_subplot(gs[68:100, :])

    # setting color
    n_model = len(roc_element_dict)
    cm_subsection = np.linspace(0, 1, n_model)
    cmap = cm.get_cmap(_DEFAULT_COLOR_PALETTE)
    colors = [cmap(x) for x in cm_subsection]

    # setting label size:
    axis_label_size = int(min(figure_size) * 1.6)
    title_label_size = int(min(figure_size) * 2.1)
    line_width = int(min(figure_size) / 6)

    ax1.set_xlabel('Thresholds', fontsize=axis_label_size)
    ax0.set_ylabel('TPR & FPR', fontsize=axis_label_size)
    ax1.set_ylabel('Margin(TPR - FPR)', fontsize=axis_label_size)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    ax0.set_xlim([0.0, 1.0])
    ax1.set_xlim([0.0, 1.0])
    ax0.set_ylim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax0.grid(True)

    plt.setp(ax0.set_xticks([]))
    # plt.setp(ax0.get_xticklabels(), visible=False)
    plt.subplots_adjust(hspace=.0)

    _index = -1
    for model_name, series_dict in roc_element_dict.items():
        _index += 1
        fpr = series_dict['fpr']
        tpr = series_dict['tpr']
        thresholds = series_dict['thresholds']
        margins = tpr - fpr if 'margins' not in series_dict else series_dict['margins']
        max_margin = margins.max()
        max_margin_threshold = thresholds[margins.argmax()]

        ax0.plot(thresholds, fpr, lw=line_width, linestyle='-', alpha=1,
                 label=f"Model {model_name} with TPR = {str(round(tpr[margins.argmax()], 3))} "
                       f"and FPR = {str(round(fpr[margins.argmax()], 3))} at max_margin margins threshold.",
                 color=colors[_index])  # fpr line;
        ax0.plot(thresholds, tpr, lw=line_width, linestyle='-', alpha=1, color=colors[_index])  # tpr line;

        ax0.axvline(x=max_margin_threshold, lw=line_width * 1.5, linestyle='-.', alpha=0.7,
                    color=colors[_index])

        # plot margins with thresholds:
        ax1.plot(thresholds, margins, lw=line_width, linestyle='-', alpha=1,
                 label=f'Model {model_name} (Max.Margin = {str(round(max_margin, 3))} at {str(round(max_margin_threshold, 3))})',
                 color=colors[_index])

        ax1.axvline(x=max_margin_threshold, lw=line_width * 1.5, linestyle='-.', alpha=0.7,
                    color=colors[_index])

    ax0.legend(loc="upper right")
    ax1.legend(loc="upper right")

    if 'subtitle' in kwargs:
        title_text = f"Trade Off Plot ({kwargs['subtitle']})"
        saved_file_name = f"trade_off_{kwargs['subtitle']}.png"
    else:
        title_text = f"Trade-Off Plot"
        saved_file_name = f"trade_off.png"

    ax0.title.set_text('Trade-Off Plot')
    ax0.title.set_fontsize(title_label_size)

    if 'save' in kwargs:
        save = kwargs['save']
        if isinstance(save, str):
            plt.savefig(save)
        else:
            plt.savefig(saved_file_name)

    plt.show()


TRUE_PROB_SERIES = np.array(['F', 'F', 'F', 'T', 'F', 'T', 'T', 'F', 'T', 'F'])
PREDICTED_PROB_SERIES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
# PREDICTED_PROB_SERIES_2 = np.array([0, 0.1, 0.15, 0.7, 0.5, 0.4, 0.6, 0.8, 0.7, 1])

plot_trade_off_curves({'a': {'true': TRUE_PROB_SERIES, 'predicted_prob': PREDICTED_PROB_SERIES, 'pos_label': 'T'}})