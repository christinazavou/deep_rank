from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import label_ranking_average_precision_score, coverage_error, label_ranking_loss
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


class Evaluation(object):

    def __init__(self, outputs, predictions, targets):

        self.outputs = outputs
        self.predictions = predictions
        self.targets = targets

    def precision_recall_fscore(self, average='macro'):
        p, r, f, _ = precision_recall_fscore_support(self.targets, self.predictions, average=average)
        return round(100*p, 3), round(100*r, 3), round(100*f, 3)

    # ASSUMING OUTPUTS ARE RANKED AND A LIST UP TO K IS RETRIEVED.
    def Recall(self, precision_at, return_all=False):
        cols = np.argsort(self.outputs, 1)[:, -precision_at:]  # because argsort makes in ascending order
        rel_per_sample = np.sum(self.targets, 1)
        found_per_sample = np.zeros(self.outputs.shape[0])
        for sample, c in enumerate(cols):
            result = self.targets[sample, c]
            found_per_sample[sample] = np.sum(result == 1)
        if return_all:
            return np.round(100*found_per_sample/rel_per_sample.astype(np.float32), 3)
        return round(100 * np.mean(found_per_sample / rel_per_sample.astype(np.float32)), 3)

    # ASSUMING OUTPUTS ARE RANKED AND A LIST UP TO K IS RETRIEVED.
    def Precision(self, precision_at, return_all=False):
        cols = np.argsort(self.outputs, 1)[:, -precision_at:]  # because argsort makes in ascending order
        found_per_sample = np.zeros(self.outputs.shape[0])
        for sample, c in enumerate(cols):
            result = self.targets[sample, c]
            found_per_sample[sample] = np.sum(result == 1)
        if return_all:
            return np.round(100*found_per_sample/float(precision_at), 3)
        return round(100 * np.mean(found_per_sample / float(precision_at)), 3)

    # ASSUMING OUTPUTS ARE RANKED AND A LIST UP TO K IS RETRIEVED.
    def upper_bound(self, precision_at, return_all=False):
        all_retrieved = np.sum(self.targets, 1)
        upper_bounds = all_retrieved / float(precision_at)
        if return_all:
            return np.round(100*upper_bounds, 3)
        return round(100*np.mean(upper_bounds), 3)
    # in fact, if R@5 is 100% but P@5 is not 100% it means we retrieved all the true elements

    def ConfusionMatrix(self, heatmap_at):
        outputs = self.outputs
        targets = self.targets
        heatmap = np.zeros((outputs.shape[1], outputs.shape[1]))
        cols = np.argsort(outputs, 1)[:, -heatmap_at:]  # because argsort makes in ascending order
        for sample, c in enumerate(cols):
            for ci in c:
                if targets[sample, ci] == 0:  # ci tag is predicted but not true
                    # for each true tag:
                    for i in np.nonzero(targets[sample])[0]:
                        heatmap[ci, i] += 1
        return heatmap

    def CorrelationMatrix(self):
        mat = np.zeros((self.targets.shape[1], self.targets.shape[1]))
        for sample, t, in enumerate(self.targets):
            true_tags = np.nonzero(t)[0]
            for i in range(len(true_tags)):
                for j in range(i + 1, len(true_tags)):
                    mat[true_tags[i], true_tags[j]] += 1
        return mat


def print_matrix(mat, tag_names, title=None, out_folder=None, xlabel=None, ylabel=None):
    # df = pd.DataFrame(mat[:25, :25], index=tag_names[:25], columns=tag_names[:25])
    df = pd.DataFrame(mat, index=tag_names, columns=tag_names)
    plt.figure(figsize=(20, 10))
    plt.pcolor(df)
    plt.colorbar()
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns, rotation=70)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
        if out_folder:
            if not os.path.isdir(out_folder):
                os.mkdir(out_folder)
            plt.savefig(os.path.join(out_folder, title + '.png'), figsize=(20, 10))
            plt.close()
        else:
            plt.show()
