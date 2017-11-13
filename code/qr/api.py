import argparse
import myio
import sys
from myio import say
from utils import load_embedding_iterator, create_embedding_layer
import numpy as np
from evaluation import Evaluation
import tensorflow as tf


class QRAPI:

    def __init__(self, model_path, corpus_path, emb_path, session, layer='lstm'):
        raw_corpus = myio.read_corpus(corpus_path)
        embedding_layer = create_embedding_layer(
            n_d=10,
            embs=load_embedding_iterator(args.embeddings),
            only_words=False
        )
        # weights = myio.create_idf_weights(corpus_path, embedding_layer) # todo
        say("vocab size={}, corpus size={}\n".format(
                embedding_layer.n_V,
                len(raw_corpus)
            ))

        if layer.lower() == "lstm":
            from models import LstmQR as Model
        elif layer.lower() in ["bilstm", "bigru"]:
            from models import BiRNNQR as Model
        elif layer.lower() == "cnn":
            from models import CnnQR as Model
        elif layer.lower() == "gru":
            from models import GruQR as Model

        model = Model(args={"layer": args.layer}, embedding_layer=embedding_layer, weights=None)

        model.load_n_set_model(model_path, session)
        say("model initialized\n")

        self.model = model

        def score_func(titles, bodies, cur_sess):
            feed_dict = {
                self.model.titles_words_ids_placeholder: titles.T,  # IT IS TRANSPOSE ;)
                self.model.bodies_words_ids_placeholder: bodies.T,  # IT IS TRANSPOSE ;)
                self.model.dropout_prob: 0.,
            }
            _scores = cur_sess.run(self.model.scores, feed_dict)
            return _scores
        self.score_func = score_func
        say("scoring function compiled\n")

    def evaluate(self, data, session):
        # return for each query the labels, ranked results, and scores
        eval_func = self.score_func
        all_ranked_labels = []
        all_ranked_ids = []
        all_ranked_scores = []
        query_ids = []
        all_MAP, all_MRR, all_Pat1, all_Pat5 = [], [], [], []
        for idts, idbs, labels, pid, qids in data:
            scores = eval_func(idts, idbs, session)
            assert len(scores) == len(labels)
            ranks = (-scores).argsort()
            ranked_scores = np.array(scores)[ranks]
            ranked_labels = labels[ranks]
            ranked_ids = np.array(qids)[ranks]
            query_ids.append(pid)
            all_ranked_labels.append(ranked_labels)
            all_ranked_ids.append(ranked_ids)
            all_ranked_scores.append(ranked_scores)
            this_ev = Evaluation([ranked_labels])
            all_MAP.append(this_ev.MAP())
            all_MRR.append(this_ev.MRR())
            all_Pat1.append(this_ev.Precision(1))
            all_Pat5.append(this_ev.Precision(5))

        print 'average all ... ', sum(all_MAP)/len(all_MAP), sum(all_MRR)/len(all_MRR), sum(all_Pat1)/len(all_Pat1), sum(all_Pat5)/len(all_Pat5)
        return all_MAP, all_MRR, all_Pat1, all_Pat5, all_ranked_labels, all_ranked_ids, query_ids, all_ranked_scores

    def write_results(self, data, session, filename):
        # return for each query: q_id, candidate_id, candidate_rank, candidate_score
        f = open(filename, 'w')
        eval_func = self.score_func
        for idts, idbs, labels, pid, qids in data:
            scores = eval_func(idts, idbs, session)
            assert len(scores) == len(labels)
            ranks = (-scores).argsort()
            ranked_scores = np.array(scores)[ranks]
            ranked_ids = np.array(qids)[ranks]
            for c_rank, (c_id, c_score) in enumerate(zip(ranked_ids, ranked_scores)):
                f.write('{} _ {} {} {} _\n'.format(pid, c_id, c_rank, c_score))


def create_eval_batches(ids_corpus, data, padding_id, pad_left):
    lst = []
    for pid, qids, qlabels in data:
        titles = []
        bodies = []
        for id in [pid]+qids:
            t, b = ids_corpus[id]
            titles.append(t)
            bodies.append(b)
        titles, bodies = myio.create_one_batch(titles, bodies, padding_id, pad_left)
        lst.append((titles, bodies, np.array(qlabels, dtype="int32"), pid, qids))
    return lst


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    # all arguments are MUST
    argparser.add_argument("--model", type=str)
    argparser.add_argument("--corpus", type=str, default="")  # texts_raw_fixed file
    argparser.add_argument("--embeddings", type=str, default="")  # embeddings file
    argparser.add_argument("--test_file", type=str, default="")
    argparser.add_argument("--full_results_file", type=str, default="")  # to write in
    argparser.add_argument("--results_file", type=str, default="")  # to write in
    argparser.add_argument("--layer", type=str, default="lstm")
    args = argparser.parse_args()
    print '\n', args, '\n'

    with tf.Session() as sess:

        myqrapi = QRAPI(args.model, args.corpus, args.embeddings, sess, args.layer)

        raw_corpus = myio.read_corpus(args.corpus)
        embedding_layer = myqrapi.model.embedding_layer
        ids_corpus = myio.map_corpus(raw_corpus, embedding_layer, max_len=100)
        test = myio.read_annotations(args.test_file, K_neg=-1, prune_pos_cnt=-1)
        test = create_eval_batches(ids_corpus, test, myqrapi.model.padding_id, pad_left=not myqrapi.model.args.average)

        testmap, testmrr, testpat1, testpat5, rank_labels, rank_ids, qids, rank_scores = myqrapi.evaluate(test, sess)

        if args.full_results_file:
            with open(args.full_results_file, 'w') as f:
                for i, (_, _, labels, pid, qids) in enumerate(test):
                    print_qids_similar = [x for x, l in zip(qids, labels) if l == 1]
                    print_qids_similar = " ".join([str(x) for x in print_qids_similar])

                    print_qids_candidates = " ".join([str(x) for x in rank_ids[i]])

                    print_ranked_scores = " ".join([str(x) for x in rank_scores[i]])

                    print_ranked_labels = " ".join([str(x) for x in rank_labels[i]])

                    f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                        pid, print_qids_similar, print_qids_candidates,
                        print_ranked_scores,
                        print_ranked_labels,
                        testmap[i], testmrr[i], testpat1[i], testpat5[i]
                    ))

        if args.results_file:
            myqrapi.write_results(test, sess, args.results_file)

# after running api and getting results you can run analyzencompareresults and then printspecialcases :)
