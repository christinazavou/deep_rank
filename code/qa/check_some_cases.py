# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
import pickle
from utils import read_results_rows, read_questions, questions_index


def print_scores(results):
    MAPS, MRRS, PAT1S, PAT5S = [r[5] for r in R], [r[6] for r in R], [r[7] for r in R], [r[8] for r in results]
    # todo: can ignore if no qidssimilar
    print 'MAP: {}\tMRR: {}\tPat1: {}\tPat5: {}\n\n'.format(
        sum(MAPS)/len(MAPS), sum(MRRS)/len(MRRS), sum(PAT1S)/len(PAT1S), sum(PAT5S)/len(PAT5S))
    MAPSpcs = np.percentile(MAPS, 25), np.percentile(MAPS, 50), np.percentile(MAPS, 75)
    MRRSpcs = np.percentile(MRRS, 25), np.percentile(MRRS, 50), np.percentile(MRRS, 75)
    PAT1Spcs = np.percentile(PAT1S, 25), np.percentile(PAT1S, 50), np.percentile(PAT1S, 75)
    PAT5Spcs = np.percentile(PAT5S, 25), np.percentile(PAT5S, 50), np.percentile(PAT5S, 75)


def analyze_bad_queries(results, out_file):
    MAPS, MRRS, PAT1S, PAT5S = [], [], [], []
    qids_not_found_at_once = []
    with open(out_file, 'w') as f:
        print 'calculating and writing ...'
        f.write('Cases where no positive case was ranked at first place (but existed).\n\n')
        for q_id, q_ids_similar, q_ids_candidates, scores, labels, map_, mrr_, pat1_, pat5_ in results:
            if pat1_ < 1 and 1 in labels:
                MAPS.append(map_), MRRS.append(mrr_), PAT1S.append(pat1_), PAT5S.append(pat5_)
                qids_not_found_at_once.append(q_id)
                f.write(u'{}:\t{}\n\t\t{}\n'.format(q_id, q_idx[q_id][0], q_idx[q_id][1]).encode('utf8'))
                f.write(u'\t\t----------------------------------------------------------------------------'
                        u'----------------------------------------------------------------------------'
                        u'\n'.encode('utf8'))
                for q_ranked, q_label in zip(q_ids_candidates, labels):
                    if q_label == 1:
                        assert q_ranked in q_ids_similar, ' OPA =/'
                        f.write(
                            u'\t\t{}\n\t\t{}\n'.format(q_idx[q_ranked][0].upper(), q_idx[q_ranked][1].upper()).encode(
                                'utf8'))
                    else:
                        assert q_ranked not in q_ids_similar, ' OPA =\\'
                        f.write(
                            u'\t\t{}\n\t\t{}\n'.format(q_idx[q_ranked][0].lower(), q_idx[q_ranked][1].lower()).encode(
                                'utf8'))
                f.write(u'------------------------------------------------------------------------------------------'
                        u'-------------------------------------------------------------------------------------------'
                        u'\n'.encode('utf8'))
    print 'MAP: {}\tMRR: {}\tPat1: {}\tPat5: {}\n\n'.format(
        sum(MAPS) / len(MAPS), sum(MRRS) / len(MRRS), sum(PAT1S) / len(PAT1S), sum(PAT5S) / len(PAT5S))
    return qids_not_found_at_once


def analyze_given_queries(results, qids_to_check, results_file):
    MAPS, MRRS, PAT1S, PAT5S = [], [], [], []
    with open(results_file, 'w') as f:
        print 'calculating and writing ...'
        f.write('Cases checked:\n\n')
        for q_id, q_ids_similar, q_ids_candidates, scores, labels, map_, mrr_, pat1_, pat5_ in results:
            if q_id in qids_to_check:
                MAPS.append(map_), MRRS.append(mrr_), PAT1S.append(pat1_), PAT5S.append(pat5_)
                f.write(u'{}:\t{}\n\t\t{}\n'.format(q_id, q_idx[q_id][0], q_idx[q_id][1]).encode('utf8'))
                f.write(u'\t\t----------------------------------------------------------------------------'
                        u'----------------------------------------------------------------------------'
                        u'\n'.encode('utf8'))
                for q_ranked, q_label in zip(q_ids_candidates, labels):
                    if q_label == 1:
                        assert q_ranked in q_ids_similar, ' OPA =/'
                        f.write(
                            u'\t\t{}\n\t\t{}\n'.format(q_idx[q_ranked][0].upper(), q_idx[q_ranked][1].upper()).encode(
                                'utf8'))
                    else:
                        assert q_ranked not in q_ids_similar, ' OPA =\\'
                        f.write(
                            u'\t\t{}\n\t\t{}\n'.format(q_idx[q_ranked][0].lower(), q_idx[q_ranked][1].lower()).encode(
                                'utf8'))
                f.write(u'------------------------------------------------------------------------------------------'
                        u'-------------------------------------------------------------------------------------------'
                        u'\n'.encode('utf8'))


argparser = argparse.ArgumentParser(sys.argv[0])
argparser.add_argument("--corpus_file", type=str, default="")
argparser.add_argument("--eval_file", type=str, default="")
argparser.add_argument("--results_file", type=str, default="")
argparser.add_argument("--save_ids", type=str, default="")
argparser.add_argument("--read_ids", type=str, default="")
args = argparser.parse_args()


Q = list(read_questions(args.corpus_file))
q_idx = questions_index(Q, True)
R = list(read_results_rows(args.eval_file))

print_scores(R)

if args.save_ids != "":
    ids_not_found = analyze_bad_queries(R, args.results_file)
    pickle.dump(ids_not_found, open(args.save_ids, 'wb'))
else:
    assert args.read_ids != ""
    ids_to_check = pickle.load(open(args.read_ids, 'rb'))
    analyze_given_queries(R, ids_to_check, args.results_file)



