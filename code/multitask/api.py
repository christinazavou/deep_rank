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
        elif layer.lower() == "bilstm":
            from models import BiLstmQRTP as Model
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
            if 'init_state' in self.model.__dict__:
                print 'init_state is in'
                feed_dict[self.model.init_state] = np.zeros((self.model.args.depth, 2, titles.T.shape[0], self.model.args.hidden_dim))
            _scores, _outputs, _predictions = cur_sess.run(
                [self.model.scores, self.model.output, self.model.prediction], feed_dict
            )
            return _scores, _outputs, _predictions
        self.score_func = score_func
        qaio.say("scoring function compiled\n")

    def evaluate(self, data, session):

        res = []

        outputs, predictions, targets = [], [], []

        for idts, idbs, id_labels, tags_b in data:
            cur_scores, cur_out, cur_pred = self.score_func(idts, idbs, session)

            outputs.append(cur_out)
            predictions.append(cur_pred)
            targets.append(tags_b)

            assert len(id_labels) == len(cur_scores)
            ranks = (-cur_scores).argsort()
            ranked_labels = id_labels[ranks]
            res.append(ranked_labels)

        e = QAEvaluation(res)
        MAP = round(e.MAP(), 4)
        MRR = round(e.MRR(), 4)
        P1 = round(e.Precision(1), 4)
        P5 = round(e.Precision(5), 4)
        print 'MAP MRR P@1 P@5: ', MAP, MRR, P1, P5

        outputs = np.vstack(outputs)
        predictions = np.vstack(predictions)
        targets = np.vstack(targets).astype(np.int32)  # it was dtype object

        ev = TPEvaluation(outputs, predictions, targets)
        # results = [round(ev.lr_ap_score(), 4), round(ev.lr_loss(), 4), round(ev.cov_error(), 4)]
        """------------------------------------------remove ill evaluation-------------------------------------------"""
        eval_labels = []
        for label in range(targets.shape[1]):
            if (targets[:, label] == np.ones(targets.shape[0])).any():
                eval_labels.append(label)
        print '\n{} labels out of {} will be evaluated (zero-sampled-labels removed).'.format(len(eval_labels), targets.shape[1])
        outputs, predictions, targets = outputs[:, eval_labels], predictions[:, eval_labels], targets[:, eval_labels]

        eval_samples = []
        for sample in range(targets.shape[0]):
            if (targets[sample, :] == np.ones(targets.shape[1])).any():
                eval_samples.append(sample)
        print '\n{} samples ouf of {} will be evaluated (zero-labeled-samples removed).'.format(len(eval_samples), outputs.shape[0])
        outputs, predictions, targets = outputs[eval_samples, :], predictions[eval_samples, :], targets[eval_samples, :]
        """------------------------------------------remove ill evaluation-------------------------------------------"""
        ev = TPEvaluation(outputs, predictions, targets)
        # results += [ev.precision_recall_fscore('macro'), ev.precision_recall_fscore('micro')]
        print 'MACRO: ', ev.precision_recall_fscore('macro')
        print 'MICRO: ', ev.precision_recall_fscore('micro')

        print 'P@1: {}\tP@3: {}\tP@5: {}\tP@10: {}\n'.format(ev.Precision(1), ev.Precision(3), ev.Precision(5), ev.Precision(10))
        print 'R@1: {}\tR@3: {}\tR@5: {}\tR@10: {}\n'.format(ev.Recall(1), ev.Recall(3), ev.Recall(5), ev.Recall(10))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    # all arguments are MUST
    argparser.add_argument("--corpus", type=str, default="")  # texts_raw_fixed file
    argparser.add_argument("--corpus_w_tags", type=str, default="")  # texts_raw_fixed_with_tags file
    argparser.add_argument("--embeddings", type=str, default="")  # embeddings file
    argparser.add_argument("--dev", type=str, default="")
    argparser.add_argument("--test", type=str, default="")
    argparser.add_argument("--tags_file", type=str, default="")
    argparser.add_argument("--model", type=str)
    argparser.add_argument("--layer", type=str, default="lstm")
    args = argparser.parse_args()
    print '\n', args, '\n'

    label_tags = pickle.load(open(args.tags_file, 'rb'))
    if isinstance(label_tags, dict):
        print 'from dict labels to list.'
        label_tags = label_tags.keys()
    print '\nloaded {} tags'.format(len(label_tags))

    with tf.Session() as sess:

        myqrapi = QRTPAPI(args.model, args.corpus, args.embeddings, len(label_tags), sess, args.layer)
        embedding_layer = myqrapi.model.embedding_layer

        raw_corpus_tags = tpio.read_corpus(args.corpus_w_tags, with_tags=True)
        ids_corpus_tags = tpio.map_corpus(raw_corpus_tags, embedding_layer, label_tags, max_len=myqrapi.model.args.max_seq_len)

        padding_id = embedding_layer.vocab_map["<padding>"]

        dev = qaio.read_annotations(args.dev, K_neg=-1, prune_pos_cnt=-1)
        dev = myio.create_eval_batches(ids_corpus_tags, dev, padding_id)
        myqrapi.evaluate(dev, sess)
        del dev

        test = qaio.read_annotations(args.test, K_neg=-1, prune_pos_cnt=-1)
        test = myio.create_eval_batches(ids_corpus_tags, test, padding_id)
        myqrapi.evaluate(test, sess)
