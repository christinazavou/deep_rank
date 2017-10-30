# -*- coding: utf-8 -*-

import argparse
import sys
import pickle
from utils import read_tp_results_rows
import utils


def find_bad_queries(results, bad_ids, out_file):
    with open(out_file, 'w') as f:
        f.write('Cases where not all true tags where retrieved up to rank 10.\n\n')
        for query_id, real_tags, rankedat10_tags, Pat5, Pat10, Rat5, Rat10, UB5, UB10 in results:
            if query_id in bad_ids:
                # print real tags based on selected tags only
                f.write(
                    u'{}:\t{}\n\t\t{}\n\t\t{}\n\t\t{}\n\n'.format(
                        query_id, q_idx[query_id][0], q_idx[query_id][1], real_tags, rankedat10_tags
                    ).encode('utf8')
                )


def find_best_queries(results, best_ids, out_file):
    with open(out_file, 'w') as f:
        f.write('Cases where all positive cases were found up to rank 5.\n\n')
        for query_id, real_tags, rankedat10_tags, Pat5, Pat10, Rat5, Rat10, UB5, UB10 in results:
            if query_id in best_ids:
                # print real tags based on selected tags only
                f.write(
                    u'{}:\t{}\n\t\t{}\n\t\t{}\n\t\t{}\n\n'.format(
                        query_id, q_idx[query_id][0], q_idx[query_id][1], real_tags, rankedat10_tags
                    ).encode('utf8')
                )


argparser = argparse.ArgumentParser(sys.argv[0])
argparser.add_argument("--corpus_w_tags", type=str, default="")
argparser.add_argument("--results", type=str, default="")
argparser.add_argument("--read_ids", type=str, default="")
argparser.add_argument("--output", type=str, default="")
args = argparser.parse_args()


Q = list(utils.read_questions_with_tags(args.corpus_w_tags))
q_idx = utils.questions_index_with_tags(Q, True)
R = list(read_tp_results_rows(args.results))

ids = pickle.load(open(args.read_ids, 'rb'))

find_bad_queries(R, ids['bad_queries'], args.output.replace('.txt', 'bad.txt'))
find_best_queries(R, ids['best_queries'], args.output.replace('.txt', 'best.txt'))
