import argparse
import json
import myio
import sys
from myio import say
from main import Model
import os
from utils import load_embedding_iterator, read_eval_rows, read_questions
import tensorflow as tf
import numpy as np
import gzip
import pickle
from evaluation import Evaluation


def load_model(model_path, corpus_path, emb_path):
    print("Loading model checkpoint from {}\n".format(model_path))

    assign_ops = []

    raw_corpus = myio.read_corpus(corpus_path)

    with gzip.open(model_path) as fin:
        data = pickle.load(fin)
        model_args = data['args']

        embedding_layer = myio.create_embedding_layer(
            raw_corpus,
            n_d=model_args.hidden_dim,
            cut_off=model_args.cut_off,
            embs=load_embedding_iterator(emb_path)
        )
        say("vocab size={}, corpus size={}\n".format(
            embedding_layer.n_V,
            len(raw_corpus)
        ))

        model = Model(args=model_args, embedding_layer=embedding_layer)

        params_dict = data['params_dict']
        graph = tf.get_default_graph()
        for param_name, param_value in params_dict.iteritems():
            variable = graph.get_tensor_by_name(param_name)
            assign_op = tf.assign(variable, param_value)
            assign_ops.append(assign_op)

    return assign_ops, model, embedding_layer


class QRAPI:

    def __init__(self, model_path, corpus_path, emb_path):
        self.assign_ops, self.model, self.embedding_layer = load_model(model_path, corpus_path, emb_path)

    def initialize(self, sess):
        sess.run(tf.global_variables_initializer())
        sess.run(self.assign_ops)

    def rank(self, sess, query):

        def score_batch(titles, bodies):
            _current_state = np.zeros((self.model.args.depth, 2, titles.T.shape[0], self.model.args.hidden_dim))
            _scores = sess.run(
                self.model.scores,
                feed_dict={
                    self.model.titles_words_ids_placeholder: titles.T,  # IT IS TRANSPOSE ;)
                    self.model.bodies_words_ids_placeholder: bodies.T,  # IT IS TRANSPOSE ;)
                    self.model.pairs_ids_placeholder: np.zeros((1, 100)),  # will be unused
                    self.model.dropout_prob: 0.,
                    self.model.init_state: _current_state
                }
            )
            return _scores

        if isinstance(query, str) or isinstance(query, unicode):
            query = json.loads(query)

        p_t, p_b = query["query"][0].strip().split(), query["query"][1].strip().split()

        questions_t = [self.embedding_layer.map_to_ids(p_t, filter_oov=True)]
        questions_b = [self.embedding_layer.map_to_ids(p_b, filter_oov=True)]

        for q in query["candidates"]:
            q_t, q_b = q[0].strip().split(), q[1].strip().split()

            questions_t.append(
                self.embedding_layer.map_to_ids(q_t, filter_oov=True)
            )
            questions_b.append(
                self.embedding_layer.map_to_ids(q_b, filter_oov=True)
            )

        batch_titles, batch_bodies = myio.create_one_batch(
            questions_t,
            questions_b,
            self.embedding_layer.vocab_map["<padding>"],
            not self.model.args.average
        )

        scores = score_batch(batch_titles, batch_bodies)

        assert len(scores) == batch_titles.shape[1] - 1, ' error in scores shape '

        # if ("BM25" in query) and ("ratio" in query):
        #     BM25 = query["BM25"]
        #     ratio = query["ratio"]
        #     assert len(BM25) == len(scores)
        #     assert ratio >= 0 and ratio <= 1.0
        #     scores = [ x*(1-ratio)+y*ratio for x,y in zip(scores, BM25) ]

        ranks = sorted(range(len(scores)), key=lambda i: -scores[i])
        return {"ranks": ranks, "scores": scores}


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--model_path", type=str)
    argparser.add_argument("--corpus_path", type=str, default="")
    argparser.add_argument("--emb_path", type=str, default="")
    argparser.add_argument("--dev", type=str, default="")
    argparser.add_argument("--results_path", type=str, default="")
    argparser.add_argument("--results_file", type=str, default="")
    args = argparser.parse_args()
    print '\n', args, '\n'

    myqrapi = QRAPI(args.model_path, args.corpus_path, args.emb_path)

    Q = list(read_questions(args.corpus_path))
    E = list(read_eval_rows(args.dev))

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    def label(x, y):
        return 1 if x in y else 0

    MAP, MRR, Pat1, Pat5 = [], [], [], []
    with open(os.path.join(args.results_path, args.results_file), 'w') as f:

        with tf.Session() as session:
            myqrapi.initialize(session)

            Qdict = {id: (qt, qb) for id, qt, qb in Q}
            for qid, qids_similar, qids_candidates in E:
                query_ = Qdict[qid]
                candidates = [Qdict[idc] for idc in qids_candidates]

                results = myqrapi.rank(session, {"query": query_, "candidates": candidates})

                ranked = results['ranks']
                scores = results['scores']
                scores = sorted(scores, reverse=True)

                ranked_candidates = np.array(qids_candidates)[ranked]
                labels = [label(item, qids_similar) for item in ranked_candidates]
                ev = Evaluation([labels])
                map_, mrr_, pat1_, pat5_ = ev.MAP(), ev.MRR(), ev.Precision(1), ev.Precision(5)
                MAP.append(map_), MRR.append(mrr_), Pat1.append(pat1_), Pat5.append(pat5_)

                qids_candidates = list(np.array(qids_candidates)[ranked])
                qids_similar = " ".join([str(x) for x in qids_similar])
                qids_candidates = " ".join([str(x) for x in qids_candidates])
                scores = " ".join([str(x) for x in scores])
                labels = " ".join([str(x) for x in labels])
                f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                    qid, qids_similar, qids_candidates,
                    scores,
                    labels,
                    map_, mrr_, pat1_, pat5_
                ))
    print 'final mean MAP\tMRR\tPat1\tPat5:\n           {}\t{}\t{}\t{}\n'.format(
        sum(MAP)/len(MAP), sum(MRR)/len(MRR), sum(Pat1)/len(Pat1), sum(Pat5)/len(Pat5)
    )

