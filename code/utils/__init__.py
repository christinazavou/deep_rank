import gzip
import numpy as np


def load_embedding_iterator(path):
    file_open = gzip.open if path.endswith(".gz") else open
    with file_open(path) as fin:
        for line in fin:
            line = line.strip()
            if line:
                parts = line.split()
                word = parts[0]
                vals = np.array([float(x) for x in parts[1:]])
                yield word, vals


def read_questions(q_file):
    with open(q_file) as f:
        questions = f.readlines()

        for line in questions:
            q_id, q_title, q_body = line.decode('utf-8').split(u'\t')
            # print "Qid: {}\tQt: {}\tQb: {}".format(q_id, q_title, q_body)
            yield int(q_id), q_title, q_body


def read_eval_rows(eval_file):  # test or dev
    with open(eval_file) as f:
        questions = f.readlines()

        for line in questions:
            if len(line.split('\t')) == 4:
                q_id, q_ids_similar, q_ids_candidates, q_bm25_candidates = \
                    line.decode('utf-8').split(u'\t')
                q_ids_similar = q_ids_similar.split(' ')
                q_ids_candidates = q_ids_candidates.split(' ')
                q_ids_similar = str2int_list(q_ids_similar)
                q_ids_candidates = str2int_list(q_ids_candidates)
            else:
                q_id, q_ids_similar, q_ids_candidates = \
                    line.decode('utf-8').split(u'\t')
                q_ids_similar = q_ids_similar.split(' ')
                q_ids_candidates = q_ids_candidates.split(' ')
                q_ids_similar = str2int_list(q_ids_similar)
                q_ids_candidates = str2int_list(q_ids_candidates)
            yield int(q_id), q_ids_similar, q_ids_candidates


def str2int_list(str_list):
    result = []
    try:
        result = [int(item) for item in str_list]
    except:
        # print('except: {}\n'.format(str_list))
        pass
    return result


def str2float_list(str_list):
    result = []
    try:
        result = [float(item) for item in str_list]
    except:
        # print('except: {}\n'.format(str_list))
        pass
    return result


# def read_results_rows(results_file):
#     with open(results_file) as f:
#         lines = f.readlines()
#
#         for line in lines:
#             q_id, q_ids_similar, q_ids_candidates, scores, labels, map_, mrr_, pat1_, pat5_ = \
#                 line.decode('utf-8').split(u'\t')
#             q_ids_similar = str2int_list(q_ids_similar.split(' '))
#             q_ids_candidates = str2int_list(q_ids_candidates.split(' '))
#             scores = str2float_list(scores.split(' '))
#             labels = str2int_list(labels.split(' '))
#             yield int(q_id), q_ids_similar, q_ids_candidates, scores, labels,\
#                 float(map_), float(mrr_), float(pat1_), float(pat5_)


def questions_index(questions_list, as_tuple=False):
    questions = {}
    for i in range(len(questions_list)):
        q_id, q_title, q_body = questions_list[i]
        if as_tuple:
            questions[int(q_id)] = (q_title, q_body)
        else:
            questions[int(q_id)] = "%s %s" % (q_title, q_body)
    return questions