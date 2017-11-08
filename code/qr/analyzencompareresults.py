import argparse
import sys
import numpy as np
import pickle
from utils import read_results_rows, read_questions, questions_index
import matplotlib.pyplot as plt


def plot_test_eval(results_list, name, results_list2=None):
    if name == 'MAP':
        idx = 5
    else:
        raise Exception()
    values = [r[idx] for r in results_list]
    if results_list2:
        values2 = [r[idx] for r in results_list2]
    # if results_list2:
    #     plt.plot(values, 'b.', values2, 'g.', alpha=0.5)
    # else:
    #     plt.plot(values, '.')
    # plt.title(name)
    # plt.show()
    if results_list2:
        plt.hist(values, bins=50, alpha=0.5, label='1')
        plt.hist(values2, bins=50, alpha=0.5, label='2')
        plt.legend(loc='upper right')
    else:
        plt.hist(values, bins=50)
    plt.title(name)
    plt.show()


argparser = argparse.ArgumentParser(sys.argv[0])
argparser.add_argument("--corpus", type=str, default="")
argparser.add_argument("--results1", type=str, default="")
argparser.add_argument("--results2", type=str, default="")
argparser.add_argument("--save_ids", type=str, default="")
argparser.add_argument("--read_ids", type=str, default="")
argparser.add_argument("--fig", type=str, default="")
args = argparser.parse_args()


Q = list(read_questions(args.corpus))
q_idx = questions_index(Q, True)

# Results of both given models
# one result contains:
# 0:pid, 1:ids_similar, 2:ids_candidates, 3:ranked_scores, 4:ranked_labels, 5:map, 6:mrr, 7:pat1, 8:pat5
R_1 = list(read_results_rows(args.results1))
plot_test_eval(R_1, 'MAP', results_list2=None)
if args.results2:
    R_2 = list(read_results_rows(args.results2))
    plot_test_eval(R_1, 'MAP', results_list2=R_2)

if args.read_ids:
    ids = pickle.load(open(args.read_ids, 'rb'))
else:
    # bad queries are defined based on the results of model 1 (or given from a file)
    # based on whether P@1 was not found (only care if similar candidate was given) !!
    # bad_queries = [r[0] for r in R_1 if 1 in r[4] and r[7] < 100]
    bad_queries = [r[0] for r in R_1 if r[5] <= 56 or (1 in r[4] and r[7] < 100)]
    print '\nbad_queries: {}\n'.format(bad_queries)

    # best queries are defined based on the results of model 1 (or given from a file)
    # based on whether MAP >= 75% i.e. most of the similar cases ranked first (only care if similar candidate was given)
    best_queries = [r[0] for r in R_1 if r[5] >= 75]  # ---------CAN CHANGE THIS VALUE----------
    print '\nbest_queries: {}\n'.format(best_queries)
    ids = {'bad_queries': bad_queries, 'best_queries': best_queries}
    if args.save_ids:
        pickle.dump(ids, open(args.save_ids, 'wb'))

if args.results2:
    # find the P@1 (P@5) values of bad (best) queries specified above for both model 1 and 2
    PAT1_model1 = [r[7] for r in R_1 if r[0] in ids['bad_queries']]
    PAT1_model2 = [r[7] for r in R_2 if r[0] in ids['bad_queries']]

    PAT5_model1 = [r[8] for r in R_1 if r[0] in ids['best_queries']]
    PAT5_model2 = [r[8] for r in R_2 if r[0] in ids['best_queries']]

    # plot differences (one positive difference means model 1 wins in one query)

    differences = np.array(PAT1_model1) - np.array(PAT1_model2) + 1
    # print differences
    print 'zero differences out of {}: {}\n'.format(len(differences), sum(differences == 0).astype(np.float32))
    plt.figure()
    plt.bar(range(len(differences)), sorted(differences, reverse=True))
    # plt.plot(range(len(differences)), sorted(differences, reverse=True), 'o')
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    plt.ylim([-100, 100])
    if not args.fig:
        plt.show()
    else:
        plt.savefig(args.fig.replace('.png', 'pat1diff.png'))

    differences = np.array(PAT5_model1) - np.array(PAT5_model2) + 1
    # print differences
    print 'zero differences out of {}: {}\n'.format(len(differences), sum(differences == 0).astype(np.float32))
    plt.figure()
    plt.bar(range(len(differences)), sorted(differences, reverse=True))
    # plt.plot(range(len(differences)), sorted(differences, reverse=True), 'o')
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    plt.ylim([-100, 100])
    if not args.fig:
        plt.show()
    else:
        plt.savefig(args.fig.replace('.png', 'mapdiff.png'))
