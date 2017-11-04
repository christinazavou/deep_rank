import numpy as np
import matplotlib.pyplot as plt

COLORS = ['r', 'g', 'b', 'y', 'k']
MARKERS = ['o', 's', '+', '*', '^']


def error_plot_all_metrics(mean_values, std_values, x_ticks, labels):

    colors = COLORS[:len(labels)]
    markers = MARKERS[:len(labels)]

    x_axis = range(len(mean_values[0]))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xticks(x_axis, x_ticks)
    y_min = np.min(np.array(mean_values)) - 5.
    y_max = np.max(np.array(mean_values)) + 5.
    plt.ylim([y_min, y_max])

    for mean_i, std_i, color_i, label_i, marker_i in zip(mean_values, std_values, colors, labels, markers):
        ax.errorbar(
            x_axis, mean_i, yerr=std_i, color=color_i, label=label_i, marker=marker_i, linestyle='--',
            linewidth=2, markeredgewidth=2, elinewidth=2, markersize=5
        )

    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax.legend(handles, labels, loc='upper left', numpoints=1)

    plt.show()


def error_plot_one_metric(mean_values, std_values, models, labels, metric):

    colors = COLORS[:len(labels)]
    markers = MARKERS[:len(models)]

    x_axis = range(len(models))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel(metric)
    plt.ylim([0, 100])

    for color_i, mean_values_i, std_values_i, label_i in zip(colors, mean_values, std_values, labels):
        for x, model, mean_value, std_value, marker in zip(x_axis, models, mean_values_i, std_values_i, markers):
            ax.errorbar(
                x, mean_value, yerr=std_value, color=color_i, label='{}_{}'.format(label_i, model), marker=marker
            )

    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax.legend(handles, labels, loc='upper left', numpoints=1)

    plt.show()


mean_values_ = [[28.0	,17.4	,62.0,	75.8], [34.2	,19.7	,75.5,	85.1], [34.3	,19.9	,75.5,	85.9],[33.3	,19.47	,73.49,	84.23]]
std_values_ = [[0.01,0.2,0.3,0.25], [0.05,0.6,0.3,0.04], [0,0.15,0,0], [0,0.4,0,1.2]]
error_plot_all_metrics(mean_values_, std_values_, ['P@5', 'P@10', 'R@5', 'R@10'], ['TagCombine','LogisticRegression','CNN','GRU'])


mean_values_ = [[57.54, 58.40, 59.79, 57.50, 61.50], [57, 59, 59.9, 57.1, 62.50], [58.54, 57.40, 59.5, 57.80, 61.90]]
std_values_ = [[0,0,0,1.3,0], [0,0.03,0.08,0.5,0], [0.3,0.04,0,1.3,0,0]]
error_plot_one_metric(mean_values_, std_values_, ['CNN', 'LSTM', 'GRU', 'BiLSTM', 'BiGRU'], ['QR', 'QRptTP', 'MTL'], 'MAP')
