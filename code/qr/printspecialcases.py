# -*- coding: utf-8 -*-

import argparse
import sys
import pickle
from utils import read_results_rows
import utils


def find_bad_queries(results, bad_ids, out_file):
    with open(out_file, 'w') as f:
        f.write('Cases where no positive case was ranked at first place (but existed).\n\n')
        for q_id, q_ids_similar, q_ids_candidates, scores, labels, map_, mrr_, pat1_, pat5_ in results:
            if q_id in bad_ids:
                f.write(u'{}:\t{}\n\t\t{}\n\t\t{}\n'.format(
                    q_id, q_idx[q_id][0], q_idx[q_id][1], q_idx[q_id][2]).encode('utf8'))
                f.write(u'\t\t----------------------------------------------------------------------------'
                        u'----------------------------------------------------------------------------'
                        u'\n'.encode('utf8'))
                for q_ranked, q_label in zip(q_ids_candidates, labels):
                    if q_label == 1:
                        assert q_ranked in q_ids_similar, ' OPA =/'
                        f.write(u'\t\t{}\n\t\t{}\n\t\t{}\n'.format(
                            q_idx[q_ranked][0].upper(), q_idx[q_ranked][1].upper(), str(q_idx[q_ranked][2]).upper()
                        ).encode('utf8'))
                    else:
                        assert q_ranked not in q_ids_similar, ' OPA =\\'
                        f.write(u'\t\t{}\n\t\t{}\n\t\t{}\n'.format(
                            q_idx[q_ranked][0].lower(), q_idx[q_ranked][1].lower(), str(q_idx[q_ranked][2]).lower()
                        ).encode('utf8'))
                f.write(u'------------------------------------------------------------------------------------------'
                        u'-------------------------------------------------------------------------------------------'
                        u'\n'.encode('utf8'))


def find_best_queries(results, best_ids, out_file):
    with open(out_file, 'w') as f:
        f.write('Cases where at least 3 of the (existing) positive cases were found up to rank 5.\n\n')
        for q_id, q_ids_similar, q_ids_candidates, scores, labels, map_, mrr_, pat1_, pat5_ in results:
            if q_id in best_ids:
                f.write(u'{}:\t{}\n\t\t{}\n\t\t{}\n'.format(
                    q_id, q_idx[q_id][0], q_idx[q_id][1], q_idx[q_id][2]).encode('utf8'))
                f.write(u'\t\t----------------------------------------------------------------------------'
                        u'----------------------------------------------------------------------------'
                        u'\n'.encode('utf8'))
                for q_ranked, q_label in zip(q_ids_candidates, labels):
                    if q_label == 1:
                        assert q_ranked in q_ids_similar, ' OPA =/'
                        f.write(u'\t\t{}\n\t\t{}\n\t\t{}\n'.format(
                            q_idx[q_ranked][0].upper(), q_idx[q_ranked][1].upper(), str(q_idx[q_ranked][2]).upper()
                        ).encode('utf8'))
                    else:
                        assert q_ranked not in q_ids_similar, ' OPA =\\'
                        f.write(u'\t\t{}\n\t\t{}\n\t\t{}\n'.format(
                            q_idx[q_ranked][0].lower(), q_idx[q_ranked][1].lower(), str(q_idx[q_ranked][2]).lower()
                        ).encode('utf8'))
                f.write(u'------------------------------------------------------------------------------------------'
                        u'-------------------------------------------------------------------------------------------'
                        u'\n'.encode('utf8'))


argparser = argparse.ArgumentParser(sys.argv[0])
argparser.add_argument("--corpus_w_tags", type=str, default="")
argparser.add_argument("--results", type=str, default="")
argparser.add_argument("--read_ids", type=str, default="")
argparser.add_argument("--output", type=str, default="")
args = argparser.parse_args()


Q = list(utils.read_questions_with_tags(args.corpus_w_tags))
q_idx = utils.questions_index_with_tags(Q, True)
R = list(read_results_rows(args.results))

ids = pickle.load(open(args.read_ids, 'rb'))

find_bad_queries(R, ids['bad_queries'], args.output.replace('.txt', 'bad.txt'))
find_best_queries(R, ids['best_queries'], args.output.replace('.txt', 'best.txt'))
