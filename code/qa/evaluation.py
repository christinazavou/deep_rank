import numpy as np


class Evaluation(object):

    def __init__(self, data):
        self.data = data

    def Precision(self, precision_at):
        scores = []
        for item in self.data:
            temp = item[:precision_at]

            relevant = 0.
            for rank in temp:
                if rank == 1:
                    relevant += 1.
            scores.append(relevant*1.0/len(temp))

        return round(100 * (sum(scores)/len(scores)), 3)

    def MAP(self):
        scores = []

        for item in self.data:

            precision_at_r = np.zeros(len(item))
            relevant = 0.
            for rank, score in enumerate(item):
                if score == 1:
                    relevant += 1.
                    precision_at_r[rank] = relevant / (rank + 1)
            if relevant == 0.:
                scores.append(0.)
            else:
                scores.append(precision_at_r.sum() / relevant)

        return round(100 * (sum(scores)/len(scores) if len(scores) > 0 else 0.0), 3)

    def MRR(self):

        scores = []
        for item in self.data:
            for rank, score in enumerate(item):
                if score == 1:
                    scores.append(1./(rank+1))
                    break
        return round(100 * (sum(scores)/len(scores) if len(scores) > 0 else 0.0), 3)
