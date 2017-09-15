import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import label_ranking_average_precision_score, coverage_error, label_ranking_loss


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
        return p, r, f

    #     # true positives
    #     tp = np.logical_and(predictions.astype(np.bool), target.astype(np.bool))
    #     # false positives
    #     fp = np.logical_and(predictions.astype(np.bool), np.logical_not(target.astype(np.bool)))
    #     # false negatives
    #     fn = np.logical_and(np.logical_not(predictions.astype(np.bool)), target.astype(np.bool))
    #
    #     if np.sum(np.logical_or(tp, fp).astype(np.int32)) == 0:
    #         print 'zero here'
    #     if np.sum(np.logical_or(tp, fn).astype(np.int32)) == 0:
    #         print 'zero there'
    #     pre = np.true_divide(np.sum(tp.astype(np.int32)), np.sum(np.logical_or(tp, fp).astype(np.int32)))
    #     rec = np.true_divide(np.sum(tp.astype(np.int32)), np.sum(np.logical_or(tp, fn).astype(np.int32)))
    #
    #     return round(pre, 4), round(rec, 4)
    #
    # def one_error(self):
    #     cols = np.argmax(self.outputs, 1)  # top ranked
    #     rows = range(self.outputs.shape[0])
    #     result = self.targets[rows, cols]
    #     return np.sum((result == 0).astype(np.int32))
    #
    # def precision_at_k(self, k):
    #     cols = np.argsort(self.outputs, 1)[:, :k]
    #     rows = range(self.outputs.shape[0])
    #     rel_per_sample = np.clip(np.sum(self.targets, 1), 0, k)
    #     found_per_sample = np.zeros(self.outputs.shape[0])
    #     for sample, c in enumerate(cols):
    #         result = self.targets[rows, cols]
    #         found_per_sample[sample] = np.sum(result == 1)
    #     return np.mean(found_per_sample / rel_per_sample.astype(np.float32))
    #
    # def accuracy(self):
    #     correct_prediction = np.equal(self.predictions, self.target)
    #     accuracy = np.mean(correct_prediction)
    #     return accuracy
