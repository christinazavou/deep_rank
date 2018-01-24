import numpy as np
import matplotlib.pyplot as plt

params = {'legend.fontsize': 'x-large',
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}
plt.rcParams.update(params)


COLORS = ['r', 'g', 'b', 'y', 'k']
MARKERS = ['o', 's', '+', '*', '^']


def error_plot_all_metrics(mean_values, std_values, x_ticks, labels, title=None, sinplin=5.):

    colors = COLORS[:len(labels)]
    markers = MARKERS[:len(labels)]

    x_axis = range(len(mean_values[0]))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xticks(x_axis, x_ticks)
    y_min = np.min(np.array(mean_values)) - sinplin
    y_max = np.max(np.array(mean_values)) + sinplin
    plt.ylim([y_min, y_max])

    if title:
        plt.title(title)

    for mean_i, std_i, color_i, label_i, marker_i in zip(mean_values, std_values, colors, labels, markers):
        ax.errorbar(
            x_axis, mean_i, yerr=std_i, color=color_i, label=label_i, marker=marker_i, linestyle='--',
            linewidth=2, markeredgewidth=2, elinewidth=2, markersize=5
        )

    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax.legend(handles, labels, loc='upper left', numpoints=1)

    plt.show()


"""-------------------------------------------------TP-----------------------------------------------------------"""
mean_values_ = [
    [27.99,	17.43,	61.96,	75.79, 45.45],  # TagCombine
    [34.23,	19.72,	75.52,	85.12,  65.58],  # LogReg
    [34.27,	19.90,	75.49,	85.92,	66.69],  # CNN
    [33.45,	19.50,	73.68,	84.17,  65.68],  # LSTM
    [33.34,	19.47,	73.67,	84.23,	65.48],  # GRU
]
std_values_ = [
    [0, 0, 0, 0, 0],  # TagCombine
    [0, 0, 0, 0, 0],  # LogReg
    [0.03,	0.01,	0.03,	0.11, 0.05],  # CNN
    [0.04,  0.02,   0.05,   0.06, 0.04],  # LSTM
    [0.03,	0.01,	0.16,	0.06, 0.06],  # GRU
]
x_ticks_ = ['P@5', 'P@10', 'R@5', 'R@10', 'MAP']
labels_ = ['TagCombine', 'LogisticRegression', 'CNN', 'LSTM', 'GRU']
title_ = 'Predefined Split'
error_plot_all_metrics(mean_values_, std_values_, x_ticks_, labels_, title_)
"""-------------------------------------------------TP-----------------------------------------------------------"""


"""-------------------------------------------------QR-----------------------------------------------------------"""
mean_values_ = [
    [56.63,	71.13,	56.99,	43.60],  # CNN
    [57.85,	70.19,	55.38,	44.49],  # LSTM
    [59.80,	72.58,	59.41,	45.48],  # GRU
    [],  # BiLSTM
    [],  # BiGRU
]
std_values_ = [
    [0.57,	0.18,	0.38,	0.51],  # CNN
    [0.47,	0.99,	1.20,	0.41],  # LSTM
    [0.57,	0.99,	2.54,	0.49],  # GRU
    [],  # BiLSTM
    [],  # BiGRU
]
x_ticks_ = ['MAP', 'MRR', 'P@1', 'P@10']
labels_ = ['CNN', 'LSTM', 'GRU', 'BiLSTM', 'BiGRU']
title_ = 'Mean Average Precision'
error_plot_all_metrics(mean_values_, std_values_, x_ticks_, labels_, title_)
"""-------------------------------------------------QR-----------------------------------------------------------"""


"""-----------------------------------------------QR various---------------------------------------------------------"""
x_ticks_ = ['QR', 'QRptTP', 'MTL']
labels_ = ['CNN', 'LSTM', 'GRU', 'BiLSTM', 'BiGRU']
title_ = 'Mean Average Precision'
"""-----------------------------------------------QR various---------------------------------------------------------"""
