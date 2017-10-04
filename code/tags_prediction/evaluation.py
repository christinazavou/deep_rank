from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import label_ranking_average_precision_score, coverage_error, label_ranking_loss
import numpy as np


class Evaluation(object):

    def __init__(self, outputs, predictions, targets):

        self.outputs = outputs
        self.predictions = predictions
        self.targets = targets

    def lr_ap_score(self):
        return label_ranking_average_precision_score(self.targets, self.outputs)

    def cov_error(self):
        return coverage_error(self.targets, self.outputs)

    def lr_loss(self):
        return label_ranking_loss(self.targets, self.outputs)

    def precision_recall_fscore(self, average='macro'):
        p, r, f, _ = precision_recall_fscore_support(self.targets, self.predictions, average=average)
        return round(100*p, 3), round(100*r, 3), round(100*f, 3)

    # ASSUMING OUTPUTS ARE RANKED AND A LIST UP TO K IS RETRIEVED.
    def Recall(self, precision_at):
        cols = np.argsort(self.outputs, 1)[:, -precision_at:]  # because argsort makes in ascending order
        rel_per_sample = np.sum(self.targets, 1)
        found_per_sample = np.zeros(self.outputs.shape[0])
        for sample, c in enumerate(cols):
            result = self.targets[sample, c]
            found_per_sample[sample] = np.sum(result == 1)
        return round(100 * np.mean(found_per_sample / rel_per_sample.astype(np.float32)), 3)

    # ASSUMING OUTPUTS ARE RANKED AND A LIST UP TO K IS RETRIEVED.
    def Precision(self, precision_at):
        cols = np.argsort(self.outputs, 1)[:, -precision_at:]  # because argsort makes in ascending order
        found_per_sample = np.zeros(self.outputs.shape[0])
        for sample, c in enumerate(cols):
            result = self.targets[sample, c]
            found_per_sample[sample] = np.sum(result == 1)
        return round(100 * np.mean(found_per_sample / float(precision_at)), 3)
