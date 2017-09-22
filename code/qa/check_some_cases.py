# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
import pickle
from utils import read_results_rows, read_questions, questions_index
import utils
import matplotlib.pyplot as plt
from collections import Counter


def print_scores(results):
    # if no similar q exists skip the query
    MAPS = [r[5] for r in R if 1 in r[4]]
    MRRS = [r[6] for r in R if 1 in r[4]]
    PAT1S = [r[7] for r in R if 1 in r[4]]
    PAT5S = [r[8] for r in results if 1 in r[4]]
    print 'MAP: {}\tMRR: {}\tPat1: {}\tPat5: {}\n\n'.format(
        sum(MAPS)/len(MAPS), sum(MRRS)/len(MRRS), sum(PAT1S)/len(PAT1S), sum(PAT5S)/len(PAT5S))
    MAPSpcs = np.percentile(MAPS, 25), np.percentile(MAPS, 50), np.percentile(MAPS, 75)
    MRRSpcs = np.percentile(MRRS, 25), np.percentile(MRRS, 50), np.percentile(MRRS, 75)
    PAT1Spcs = np.percentile(PAT1S, 25), np.percentile(PAT1S, 50), np.percentile(PAT1S, 75)
    PAT5Spcs = np.percentile(PAT5S, 25), np.percentile(PAT5S, 50), np.percentile(PAT5S, 75)


def find_bad_queries(results, out_file):
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


def find_best_queries(results, out_file):
    MAPS, MRRS, PAT1S, PAT5S = [], [], [], []
    qids_ranked_well = []
    with open(out_file, 'w') as f:
        print 'calculating and writing ...'
        f.write('Cases where no positive case was ranked at first place (but existed).\n\n')
        for q_id, q_ids_similar, q_ids_candidates, scores, labels, map_, mrr_, pat1_, pat5_ in results:
            if pat5_ > 0.45 and labels.count(1) > 5:
                MAPS.append(map_), MRRS.append(mrr_), PAT1S.append(pat1_), PAT5S.append(pat5_)
                qids_ranked_well.append(q_id)
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
    return qids_ranked_well


def analyze_given_queries(results, qids_to_check, special_cases_file):
    MAPS, MRRS, PAT1S, PAT5S = [], [], [], []
    with open(special_cases_file, 'w') as f:
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
    print 'MAP: {}\tMRR: {}\tPat1: {}\tPat5: {}\n\n'.format(
        sum(MAPS) / len(MAPS), sum(MRRS) / len(MRRS), sum(PAT1S) / len(PAT1S), sum(PAT5S) / len(PAT5S))


def good_vs_bad_statistics(results):
    assert isinstance(q_idx.values()[0], tuple) and len(q_idx.values()[0]) == 2,  q_idx.values()[0]
    q_indicators = ["what", "when", "where", "why", "how", "who"]
    bad_words_difference, best_words_difference = [], []
    bad_q_indicators_difference, best_q_indicators_difference = [], []
    bad_common_tags_count, best_common_tags_count = [], []
    bad_tags, best_tags = Counter(), Counter()
    for q_id, q_ids_similar, q_ids_candidates, scores, labels, map_, mrr_, pat1_, pat5_ in results:
        if pat1_ < 1 and 1 in labels:
            query, q_tags = q_idx[q_id]
            query_indicators = [q_ind for q_ind in q_indicators if q_ind in query.split(' ')]
            bad_tags += Counter(q_tags)
            for p_id in q_ids_similar:
                candidate, c_tags = q_idx[p_id]
                bad_words_difference.append(abs(len(query.split(' ')) - len(candidate.split(' '))))
                candidate_indicators = [q_ind for q_ind in q_indicators if q_ind in candidate.split(' ')]
                bad_q_indicators_difference.append(abs(len(query_indicators) - len(candidate_indicators)))
                bad_tags += Counter(c_tags)
                bad_common_tags_count.append(len(set(q_tags) & set(c_tags)))
        if pat5_ > 0.54 and labels.count(1) > 5:
            query, q_tags = q_idx[q_id]
            query_indicators = [q_ind for q_ind in q_indicators if q_ind in query.split(' ')]
            best_tags += Counter(q_tags)
            for p_id in q_ids_similar:
                candidate, c_tags = q_idx[p_id]
                best_words_difference.append(abs(len(query.split(' ')) - len(candidate.split(' '))))
                candidate_indicators = [q_ind for q_ind in q_indicators if q_ind in candidate.split(' ')]
                best_q_indicators_difference.append(abs(len(query_indicators) - len(candidate_indicators)))
                best_tags += Counter(c_tags)
                best_common_tags_count.append(len(set(q_tags) & set(c_tags)))

    print 'bad words difference mean, std = ', np.mean(np.array(bad_words_difference)), np.std(np.array(bad_words_difference))
    print 'best words difference mean, std = ', np.mean(np.array(best_words_difference)), np.std(np.array(best_words_difference))
    print 'bad q indicators difference mean, std = ', np.mean(np.array(bad_q_indicators_difference)), np.std(np.array(bad_q_indicators_difference))
    print 'best q indicators difference mean, std = ', np.mean(np.array(best_q_indicators_difference)), np.std(np.array(best_q_indicators_difference))
    print 'bad common tags count mean, std = ', np.mean(np.array(bad_common_tags_count)), np.std(np.array(bad_common_tags_count))
    print 'best common tags count mean, std = ', np.mean(np.array(best_common_tags_count)), np.std(np.array(best_common_tags_count))
    # plt.figure()
    # plt.hist(bad_words_difference)
    # plt.title('amount of words difference in bad')
    # plt.figure()
    # plt.hist(best_words_difference)
    # plt.title('amount of words difference in best')
    # plt.figure()
    # plt.hist(bad_q_indicators_difference)
    # plt.title('question-indicators difference in bad')
    # plt.figure()
    # plt.hist(best_q_indicators_difference)
    # plt.title('question-indicators difference in best')
    # plt.figure()
    # plt.hist(bad_common_tags_count)
    # plt.title('count of common tags in bad')
    # plt.figure()
    # plt.hist(best_common_tags_count)
    # plt.title('count of common tags in best')
    # plt.show()
    print 'bad tags: ', bad_tags, '\n'
    print 'best tags: ', best_tags, '\n'

argparser = argparse.ArgumentParser(sys.argv[0])
argparser.add_argument("--corpus", type=str, default="")
argparser.add_argument("--corpus_w_tags", type=str, default="")
argparser.add_argument("--results", type=str, default="")
argparser.add_argument("--bad_cases", type=str, default="")
argparser.add_argument("--best_cases", type=str, default="")
argparser.add_argument("--save_bad_ids", type=str, default="")
argparser.add_argument("--given_ids", type=str, default="")
argparser.add_argument("--analyzed_ids", type=str, default="")
args = argparser.parse_args()


Q = list(read_questions(args.corpus))
q_idx = questions_index(Q, True)
R = list(read_results_rows(args.results))

print_scores(R)

if args.bad_cases:
    ids_not_found = find_bad_queries(R, args.bad_cases)
    if args.save_bad_ids:
        pickle.dump(ids_not_found, open(args.save_ids, 'wb'), protocol=2)

if args.best_cases:
    find_best_queries(R, args.best_cases)

if args.given_ids:
    ids_to_check = pickle.load(open(args.read_ids, 'rb'))
    analyze_given_queries(R, ids_to_check, args.analyzed_ids)

if args.corpus_w_tags:
    Q = list(utils.read_questions_with_tags(args.corpus_w_tags))
    q_idx = utils.questions_index_with_tags(Q, False)
    good_vs_bad_statistics(R)


