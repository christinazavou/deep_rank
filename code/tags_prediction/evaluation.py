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
        return round(p, 4), round(r, 4), round(f, 4)

    # def precision_recall(self):
    #     # true positives
    #     tp = np.logical_and(self.predictions.astype(np.bool), self.targets.astype(np.bool))
    #     # false positives
    #     fp = np.logical_and(self.predictions.astype(np.bool), np.logical_not(self.targets.astype(np.bool)))
    #     # false negatives
    #     fn = np.logical_and(np.logical_not(self.predictions.astype(np.bool)), self.targets.astype(np.bool))
    #     if np.sum(np.logical_or(tp, fp).astype(np.int32)) == 0:
    #         print 'zero tp and fp i.e. predictions'
    #     if np.sum(np.logical_or(tp, fn).astype(np.int32)) == 0:
    #         print 'zero tp and fn i.e. true targets'
    #     pre = np.true_divide(np.sum(tp.astype(np.int32)), np.sum(np.logical_or(tp, fp).astype(np.int32)))
    #     rec = np.true_divide(np.sum(tp.astype(np.int32)), np.sum(np.logical_or(tp, fn).astype(np.int32)))
    #     return round(pre, 4), round(rec, 4)

    # def accuracy(self):
    #     correct_prediction = np.equal(self.predictions, self.targets)
    #     accuracy = np.mean(correct_prediction)
    #     return accuracy

    # ASSUMING OUTPUTS ARE RANKED AND A LIST UP TO K IS RETRIEVED.
    def Recall(self, precision_at):
        cols = np.argsort(self.outputs, 1)[:, -precision_at:]  # because argsort makes in ascending order
        rel_per_sample = np.clip(np.sum(self.targets, 1), 0, precision_at)
        found_per_sample = np.zeros(self.outputs.shape[0])
        for sample, c in enumerate(cols):
            result = self.targets[sample, c]
            found_per_sample[sample] = np.sum(result == 1)
        return np.mean(found_per_sample / rel_per_sample.astype(np.float32))

    # ASSUMING OUTPUTS ARE RANKED AND A LIST UP TO K IS RETRIEVED.
    def Precision(self, precision_at):
        cols = np.argsort(self.outputs, 1)[:, -precision_at:]  # because argsort makes in ascending order
        found_per_sample = np.zeros(self.outputs.shape[0])
        for sample, c in enumerate(cols):
            result = self.targets[sample, c]
            found_per_sample[sample] = np.sum(result == 1)
        return np.mean(found_per_sample / float(precision_at))

    # todo: should input a list of 0 and 1
    # # ASSUMING OUTPUTS ARE RANKED AND A LIST UP TO K IS RETRIEVED.
    # def DCG(self, ndcg_at):
    #     cols = np.argsort(self.outputs, 1)[:, -ndcg_at:]  # because argsort makes in ascending order
    #     ndcg_per_sample = np.zeros(self.outputs.shape[0])
    #     for sample, c in enumerate(cols):
    #         ndcg_per_sample[sample] = self.targets[sample, c[0]]
    #         for rank, retrieved_relevance in enumerate(self.targets[sample, c]):
    #             if rank == 0:  # already accounted
    #                 continue
    #             if retrieved_relevance != 0:
    #                 ndcg_per_sample[sample] += retrieved_relevance / np.log2((rank + 1 + 1))
    #                 # dcg that puts more emphasis on retrieving relevant docs:
    #                 # dcg += (math.pow(2, retrieved_relevance) - 1.) / np.log2((rank + 1 + 1))  # in binary relevance no diff !
    #     return ndcg_per_sample
    #
    # def ideal_rank_relevances(self, ndcg_at):
    #     rel_per_sample = np.clip(np.sum(self.targets, 1), 0, ndcg_at)
    #     ideal_ranks = np.zeros((self.targets.shape[0], ndcg_at))
    #     for sample, rel_num in enumerate(rel_per_sample):
    #         ideal_ranks[sample, 0:rel_num] = 1
    #     return ideal_ranks
    #
    # def NDCG(self, ndcg_at):
    #     return np.divide(self.DCG(ndcg_at), self.DCG(self.ideal_rank_relevances(ndcg_at)))
