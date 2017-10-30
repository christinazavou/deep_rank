import argparse
import sys
import numpy as np
import pickle
from utils import read_tp_results_rows, read_questions_with_tags, questions_index_with_tags
import matplotlib.pyplot as plt


argparser = argparse.ArgumentParser(sys.argv[0])
argparser.add_argument("--corpus_w_tags", type=str, default="")
argparser.add_argument("--results1", type=str, default="")
argparser.add_argument("--results2", type=str, default="")
argparser.add_argument("--save_ids", type=str, default="")
argparser.add_argument("--read_ids", type=str, default="")
argparser.add_argument("--fig", type=str, default="")
args = argparser.parse_args()


Q = list(read_questions_with_tags(args.corpus_w_tags))
q_idx = questions_index_with_tags(Q, True)

# Results of both given models
# one result contains:
# 0:query_id, 1:real_tags, 2:rankedat10_tags, 3:Pat5, 4:Pat10, 5:Rat5, 6:Rat10, 7:UpperBound5, 8:UpperBound10
R_1 = list(read_tp_results_rows(args.results1))
if args.results2:
    R_2 = list(read_tp_results_rows(args.results2))

if args.read_ids:
    ids = pickle.load(open(args.read_ids, 'rb'))
else:
    # bad queries are defined based on the results of model 1 (or given from a file)
    # based on whether R@10 was < 50%
    # no need to test if r[1] != [''] since they are not considered in evaluation (& thus in results)
    bad_queries = [r[0] for r in R_1 if r[6] < 50]  # ---------CAN CHANGE THIS VALUE----------
    print '\nbad_queries: {}\n'.format(bad_queries)

    # best queries are defined based on the results of model 1 (or given from a file)
    # based on whether all true positives where found up to rank 5 (use of upper bound)
    # no need to test if r[1] != [''] since they are not considered in evaluation (& thus in results)
    best_queries = [r[0] for r in R_1 if r[5] == 100]
    print '\nbest_queries: {}\n'.format(best_queries)
    ids = {'bad_queries': bad_queries, 'best_queries': best_queries}
    if args.save_ids:
        pickle.dump(ids, open(args.save_ids, 'wb'))

if args.results2:
    # find R@5 (R@10) values of best (bad) queries specified above for both model 1 and 2
    RAT5_model1 = [r[5] for r in R_1 if r[0] in ids['best_queries']]
    RAT5_model2 = [r[5] for r in R_2 if r[0] in ids['best_queries']]

    RAT10_model1 = [r[6] for r in R_1 if r[0] in ids['bad_queries']]
    RAT10_model2 = [r[6] for r in R_2 if r[0] in ids['bad_queries']]

    # plot differences (one positive difference means model 1 wins in one query)

    differences = np.array(RAT10_model1) - np.array(RAT10_model1)
    # print differences
    print 'zero differences out of {}: {}\n'.format(len(differences), sum(differences == 0).astype(np.float32))
    plt.figure()
    plt.bar(range(len(differences)), sorted(differences, reverse=True))
    # plt.plot(differences, '.')
    plt.ylim([-100, 100])
    plt.savefig(args.fig.replace('.png', 'rat10diff.png'))

    differences = np.array(RAT5_model1) - np.array(RAT5_model2)
    # print differences
    print 'zero differences out of {}: {}\n'.format(len(differences), sum(differences == 0).astype(np.float32))
    plt.figure()
    # plt.plot(differences, '.')
    plt.bar(range(len(differences)), sorted(differences, reverse=True))
    plt.ylim([-100, 100])
    plt.savefig(args.fig.replace('.png', 'rat5diff.png'))
