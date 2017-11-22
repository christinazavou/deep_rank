import argparse
import myio
import sys
from utils import load_embedding_iterator, create_embedding_layer
import numpy as np
import tensorflow as tf
from qr import myio as qaio
from tags_prediction import myio as tpio
import pickle
from qr.evaluation import Evaluation as QAEvaluation
from tags_prediction.evaluation import Evaluation as TPEvaluation


class QRTPAPI:

    def __init__(self, model_path, corpus_path, emb_path, output_dim, session, layer):
        raw_corpus = qaio.read_corpus(corpus_path)
        embedding_layer = create_embedding_layer(
            n_d=10,
            embs=load_embedding_iterator(args.embeddings),
            only_words=False
        )
        # weights = myio.create_idf_weights(corpus_path, embedding_layer) # todo
        qaio.say("vocab size={}, corpus size={}\n".format(
            embedding_layer.n_V,
            len(raw_corpus)
        ))

        if layer.lower() == "lstm":
            from models import LstmQRTP as Model
        elif layer.lower() in ["bilstm", "bigru"]:
            from models import BiRNNQRTP as Model
            raise Exception()
        elif layer.lower() == "cnn":
            from models import CnnQRTP as Model
        elif layer.lower() == "gru":
            from models import GruQRTP as Model

        # model = Model(args=None, embedding_layer=embedding_layer, output_dim=output_dim, weights=weights)
        model = Model(args=None, embedding_layer=embedding_layer, output_dim=output_dim)

        model.load_n_set_model(model_path, session)
        qaio.say("model initialized\n")

        self.model = model

        def score_func(titles, bodies, cur_sess):
            feed_dict = {
                self.model.titles_words_ids_placeholder: titles.T,  # IT IS TRANSPOSE ;)
                self.model.bodies_words_ids_placeholder: bodies.T,  # IT IS TRANSPOSE ;)
                self.model.dropout_prob: 0.,
            }
            _scores, _outputs = cur_sess.run(
                [self.model.scores, self.model.act_output], feed_dict
            )
            return _scores, _outputs
        self.score_func = score_func
        qaio.say("scoring function compiled\n")

    def evaluate(self, data, session):

        res = []

        outputs, targets = [], []

        for idts, idbs, id_labels, tags_b, tuples_b, pid, qids in data:
            cur_scores, cur_out = self.score_func(idts, idbs, session)

            outputs.append(cur_out)
            targets.append(tags_b)

            assert len(id_labels) == len(cur_scores)
            ranks = (-cur_scores).argsort()
            ranked_labels = id_labels[ranks]
            res.append(ranked_labels)

        e = QAEvaluation(res)
        print '\nMAP: {} MRR: {} P@1: {} P@5: {}\n'.format(e.MAP(), e.MRR(), e.Precision(1), e.Precision(5))

        outputs = np.vstack(outputs)
        targets = np.vstack(targets).astype(np.int32)  # it was dtype object

        """------------------------------------------remove ill evaluation-------------------------------------------"""
        eval_samples = []
        for sample in range(targets.shape[0]):
            if (targets[sample, :] == np.ones(targets.shape[1])).any():
                eval_samples.append(sample)
        print '\n{} samples ouf of {} will be evaluated (zero-labeled-samples removed).'.format(len(eval_samples), outputs.shape[0])
        outputs, targets = outputs[eval_samples, :], targets[eval_samples, :]
        """------------------------------------------remove ill evaluation-------------------------------------------"""
        ev = TPEvaluation(outputs, None, targets)

        print '\naverage: P@5: {} P@10: {} R@5: {} R@10: {} UBP@5: {} UBP@10: {} MAP: {}\n'.format(
            ev.Precision(5), ev.Precision(10), ev.Recall(5), ev.Recall(10), ev.upper_bound(5), ev.upper_bound(10),
            ev.MeanAveragePrecision()
        )

    def write_results(self, data, session, filename):
        # return for each query: q_id, candidate_id, candidate_rank, candidate_score
        f = open(filename, 'w')
        eval_func = self.score_func
        for idts, idbs, labels, tags, tuples, pid, qids in data:
            scores, _ = eval_func(idts, idbs, session)
            assert len(scores) == len(labels)
            ranks = (-scores).argsort()
            ranked_scores = np.array(scores)[ranks]
            ranked_ids = np.array(qids)[ranks]
            for c_rank, (c_id, c_score) in enumerate(zip(ranked_ids, ranked_scores)):
                f.write('{} _ {} {} {} _\n'.format(pid, c_id, c_rank, c_score))


def create_eval_batches(ids_corpus, data, padding_id, N_neg=20):
    lst = []

    def transform(counter, x, length):
        return ((counter - 1) * length) + x

    for pid, qids, qlabels in data:
        titles = []
        bodies = []
        tag_labels = []
        cnt_q = 0
        tuples = []
        for id in [pid]+qids:
            cnt_q += 1
            title, body, tag = ids_corpus[str(id)]
            titles.append(title)
            bodies.append(body)
            tag_labels.append(tag)

            q_positive_ids = [transform(cnt_q, idx, tag.shape[0]) for idx, label in enumerate(tag) if label == 1]
            q_negative_ids = [transform(cnt_q, idx, tag.shape[0]) for idx, label in enumerate(tag) if label == 0]
            np.random.shuffle(q_negative_ids)
            q_negative_ids = q_negative_ids[:N_neg]  # consider only 20 negatives
            tuples += [[p_id] + q_negative_ids for p_id in q_positive_ids]

        tuples = myio.create_hinge_batch(tuples)
        titles, bodies, tag_labels = myio.create_one_batch(titles, bodies, tag_labels, padding_id)
        lst.append((titles, bodies, np.array(qlabels, dtype="int32"), tag_labels, tuples, pid, qids))

    return lst


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    # all arguments are MUST
    argparser.add_argument("--corpus", type=str, default="")  # texts_raw_fixed file
    argparser.add_argument("--corpus_w_tags", type=str, default="")  # texts_raw_fixed_with_tags file
    argparser.add_argument("--embeddings", type=str, default="")  # embeddings file
    argparser.add_argument("--test_file", type=str, default="")
    argparser.add_argument("--tags_file", type=str, default="")
    argparser.add_argument("--model", type=str)
    argparser.add_argument("--mlp_dim_tp", type=int, default=50)
    argparser.add_argument("--layer", type=str, default="lstm")
    argparser.add_argument("--results_file", type=str, default="")  # to write in
    args = argparser.parse_args()
    print '\n', args, '\n'

    label_tags = pickle.load(open(args.tags_file, 'rb'))
    print '\nloaded {} tags'.format(len(label_tags))

    with tf.Session() as sess:

        myqrapi = QRTPAPI(args.model, args.corpus, args.embeddings, len(label_tags), sess, args.layer)
        embedding_layer = myqrapi.model.embedding_layer

        raw_corpus_tags = tpio.read_corpus(args.corpus_w_tags, with_tags=True)
        ids_corpus_tags = tpio.map_corpus(raw_corpus_tags, embedding_layer, label_tags, max_len=myqrapi.model.args.max_seq_len)

        padding_id = embedding_layer.vocab_map["<padding>"]

        test = qaio.read_annotations(args.test_file, K_neg=-1, prune_pos_cnt=-1)
        test = create_eval_batches(ids_corpus_tags, test, padding_id)
        myqrapi.evaluate(test, sess)

        if args.results_file:
            myqrapi.write_results(test, sess, args.results_file)
