import numpy as np
import matplotlib.pyplot as plt

COLORS = ['r', 'g', 'b', 'y', 'k']
MARKERS = ['o', 's', '+', '*', '^']


def error_plot_all_metrics(mean_values, std_values, x_ticks, labels, title=None):

    colors = COLORS[:len(labels)]
    markers = MARKERS[:len(labels)]

    x_axis = range(len(mean_values[0]))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xticks(x_axis, x_ticks)
    y_min = np.min(np.array(mean_values)) - 5.
    y_max = np.max(np.array(mean_values)) + 5.
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

#
# def error_plot_one_metric(mean_values, std_values, models, labels, metric):
#
#     colors = COLORS[:len(labels)]
#     markers = MARKERS[:len(models)]
#
#     x_axis = range(len(models))
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_ylabel(metric)
#     plt.ylim([0, 100])
#
#     for color_i, mean_values_i, std_values_i, label_i in zip(colors, mean_values, std_values, labels):
#         for x, model, mean_value, std_value, marker in zip(x_axis, models, mean_values_i, std_values_i, markers):
#             ax.errorbar(
#                 x, mean_value, yerr=std_value, color=color_i, label='{}_{}'.format(label_i, model), marker=marker
#             )
#
#     handles, labels = ax.get_legend_handles_labels()
#     handles = [h[0] for h in handles]
#     ax.legend(handles, labels, loc='upper left', numpoints=1)
#
#     plt.show()


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


"""----------------------------------------------TP crossval--------------------------------------------------------"""
mean_values_ = [
    [24.76,	15.90,	54.37,	68.33],  # TagCombine
    [32.76,	19.14,	71.68,	81.97],  # LogReg
    [32.82,	19.21,	71.49,	82.08],  # CNN
    [31.75,	18.64,	69.75,	80.25],  # LSTM
    [31.96,	18.78,	69.79,	80.38],  # GRU
]
std_values_ = [
    [0.52,	0.10,	1.14,	0.44],  # TagCombine
    [0.05,	0.04,	0.07,	0.01],  # LogReg
    [0.03,  0.05,   0.10,   0.20],  # CNN
    [0.04,	0.02,	0.21,	0.25],  # LSTM
    [0.08,	0.04,	0.03,	0.02],  # GRU
]
x_ticks_ = ['P@5', 'P@10', 'R@5', 'R@10']
labels_ = ['TagCombine', 'LogisticRegression', 'CNN', 'LSTM', 'GRU']
title_ = 'Cross Validation'
error_plot_all_metrics(mean_values_, std_values_, x_ticks_, labels_, title_)
"""----------------------------------------------TP crossval--------------------------------------------------------"""

"""-------------------------------------------------QR-----------------------------------------------------------"""
mean_values_ = [
    [57.54,	72.20,	57.80,	44.11],  # CNN
    [57.80, 70.59,  55.51,  43.98],  # LSTM
    [59.79,	73.06,	59.46,	44.99],  # GRU
    [58.40,	72.14,	58.33,	43.01],  # BiLSTM
    [60.59,	73.35,	59.68,	45.35],  # BiGRU
]
std_values_ = [
    [0.23,	0.79,	1.54,	0.35],  # CNN
    [0.45,  0.53,   1.39,   0.42],  # LSTM
    [0.94,	0.67,	1.21,	0.79],  # GRU
    [0.99,	1.71,	2.88,	1.00],  # BiLSTM
    [1.08,	0.86,	1.82,	0.73],  # BiGRU
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
