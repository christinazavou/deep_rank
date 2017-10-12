import argparse
import myio
import sys
from qr.myio import say
from utils import load_embedding_iterator, read_df, create_embedding_layer
import numpy as np
from evaluation import Evaluation
import pickle
import tensorflow as tf
from evaluation import print_matrix
import os


class TPAPI:

    def __init__(self, model_path, emb_layer, session, output_dim, layer='lstm'):

        if layer.lower() == 'lstm':
            from models import LstmMultiTagsClassifier as Model
        elif layer.lower() == 'bilstm':
            from models import BiLstmMultiTagsClassifier as Model
        elif layer.lower() == 'cnn':
            from models import CnnMultiTagsClassifier as Model
        elif layer.lower() == "gru":
            from models import GruMultiTagsClassifier as Model

        # weights = myio.create_idf_weights(corpus_path, embedding_layer) # todo

        model = Model(args=None, embedding_layer=emb_layer, output_dim=output_dim)

        model.load_n_set_model(model_path, session)
        say("model initialized\n")

        assert model.b_o.get_shape()[0] == output_dim

        self.model = model

        def predict_func(titles, bodies, cur_sess):

            outputs, predictions = cur_sess.run(
                [self.model.act_output, self.model.prediction],
                feed_dict={
                    self.model.titles_words_ids_placeholder: titles.T,  # IT IS TRANSPOSE ;)
                    self.model.bodies_words_ids_placeholder: bodies.T,  # IT IS TRANSPOSE ;)
                    self.model.dropout_prob: 0.,
                }
            )
            return outputs, predictions

        self.predict_func = predict_func
        say("Prediction function compiled\n")

    def evaluate(self, data, tag_names, folder, session):

        eval_func = self.predict_func
        outputs, predictions, targets = [], [], []
        for idts, idbs, tags in data:
            output, prediction = eval_func(idts, idbs, session)
            outputs.append(output)
            predictions.append(prediction)
            targets.append(tags)

        outputs = np.vstack(outputs)
        predictions = np.vstack(predictions)
        targets = np.vstack(targets).astype(np.int32)  # it was dtype object

        # ev = Evaluation(outputs, predictions, targets)
        # print 'label_ranking_average_precision_score: ', round(ev.lr_ap_score(), 4)
        # print 'label_ranking_loss: ', round(ev.lr_loss(), 4)
        # print 'coverage_error: ', round(ev.cov_error(), 4)

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

        ev = Evaluation(outputs, predictions, targets)
        print 'MACRO PRECISION RECALL F1: ', ev.precision_recall_fscore(average='macro')
        print 'MICRO PRECISION RECALL F1: ', ev.precision_recall_fscore(average='micro')
        print 'P@1: {}\tP@3: {}\tP@5: {}\tP@10: {}\tR@1: {}\tR@3: {}\tR@5: {}\tR@10: {}\n'.format(
            ev.Precision(1), ev.Precision(3), ev.Precision(5), ev.Precision(10),
            ev.Recall(1), ev.Recall(3), ev.Recall(5), ev.Recall(10))

        # mat = ev.ConfusionMatrix(5)  # todo: fix for eval_labels not all labels
        # print_matrix(mat, tag_names, 'Confusion:True Tag on x-axis, False Tag on y-axis', folder)
        # mat = ev.CorrelationMatrix()
        # print_matrix(mat, tag_names, 'Correlation: True Tag on both axis', folder)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--corpus_w_tags", type=str, default="")
    argparser.add_argument("--embeddings", type=str, default="")
    argparser.add_argument("--df_corpus", type=str, default="")
    argparser.add_argument("--tags_file", type=str, default="")
    argparser.add_argument("--model", type=str)
    argparser.add_argument("--layer", type=str, default="lstm")
    argparser.add_argument("--max_seq_len", type=int, default=100)
    argparser.add_argument("--out_dir", type=str)

    args = argparser.parse_args()
    print '\n', args, '\n'

    df = read_df(args.df_corpus)
    df = df.fillna(u'')

    label_tags = pickle.load(open(args.tags_file, 'rb'))

    raw_corpus = myio.read_corpus(args.corpus_w_tags, with_tags=True)
    embedding_layer = create_embedding_layer(
        n_d=10,
        embs=load_embedding_iterator(args.embeddings),
        only_words=False
    )

    with tf.Session() as sess:

        myqrapi = TPAPI(args.model, embedding_layer, sess, len(label_tags), args.layer)

        embedding_layer = myqrapi.model.embedding_layer

        ids_corpus = myio.map_corpus(raw_corpus, embedding_layer, label_tags, max_len=args.max_seq_len)

        print("vocab size={}, corpus size={}\n".format(embedding_layer.n_V, len(raw_corpus)))

        padding_id = embedding_layer.vocab_map["<padding>"]

        say("vocab size={}, corpus size={}\n".format(
            embedding_layer.n_V,
            len(raw_corpus)
        ))

        eval_batches = myio.create_batches(df, ids_corpus, 'dev', myqrapi.model.args.batch_size, padding_id)
        print 'DEV evaluation:'
        print '{} batches.'.format(len(eval_batches))
        myqrapi.evaluate(eval_batches, label_tags, os.path.join(args.out_dir, 'dev') if args.out_dir else None, sess)

        eval_batches = myio.create_batches(df, ids_corpus, 'test', myqrapi.model.args.batch_size, padding_id)
        print 'TEST evaluation:'
        print '{} batches.'.format(len(eval_batches))
        myqrapi.evaluate(eval_batches, label_tags, os.path.join(args.out_dir, 'test') if args.out_dir else None, sess)