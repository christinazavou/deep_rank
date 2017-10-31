import argparse
import sys
import numpy as np
from evaluation import Evaluation


argparser = argparse.ArgumentParser(sys.argv[0])
argparser.add_argument("--test_file", type=str, default="")
argparser.add_argument("--results_file", type=str, default="")
args = argparser.parse_args()


all_MAP, all_MRR, all_Pat1, all_Pat5 = [], [], [], []


def label(sample, pos_samples):
    return 1 if sample in pos_samples else 0


with open(args.test_file) as fin:
    with open(args.results_file, 'w') as fout:
        for line in fin:
            pid, pos, cands, scores = line.split("\t")
            pos = pos.split()
            if len(pos) == 0:
                continue
            cands = cands.split()
            scores = scores.split()
            scores = np.array([float(score) for score in scores])

            scores_sorted_ids = scores.argsort()[::-1]
            scores_sorted = [scores[idx] for idx in scores_sorted_ids]
            cands_sorted = [cands[idx] for idx in scores_sorted_ids]
            labels_sorted = [label(cand, pos) for cand in cands_sorted]

            this_ev = Evaluation([labels_sorted])
            this_map, this_mrr, this_pat1, this_pat5 = this_ev.MAP(), this_ev.MRR(), \
                this_ev.Precision(1), this_ev.Precision(5)
            all_MAP.append(this_map)
            all_MRR.append(this_mrr)
            all_Pat1.append(this_pat1)
            all_Pat5.append(this_pat5)

            pos = " ".join([str(x) for x in pos])
            cands = " ".join([str(x) for x in cands])
            scores_sorted = " ".join([str(x) for x in scores_sorted])
            labels_sorted = " ".join([str(x) for x in labels_sorted])

            fout.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                pid, pos, cands, scores_sorted, labels_sorted,
                this_map, this_mrr, this_pat1, this_pat5
            ))

print 'average all ... ', sum(all_MAP)/len(all_MAP), sum(all_MRR)/len(all_MRR), sum(all_Pat1)/len(all_Pat1), \
    sum(all_Pat5)/len(all_Pat5)

