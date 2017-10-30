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
import random


class TPAPI:

    def __init__(self, model_path, emb_layer, session, output_dim, layer='lstm'):

        if layer.lower() == 'lstm':
            from models import LstmMultiTagsClassifier as Model
        elif layer.lower() in ['bilstm', "bigru"]:
            from models import BiRNNMultiTagsClassifier as Model
        elif layer.lower() == 'cnn':
            from models import CnnMultiTagsClassifier as Model
        elif layer.lower() == "gru":
            from models import GruMultiTagsClassifier as Model

        model = Model(args=None, embedding_layer=emb_layer, output_dim=output_dim, weights=None)

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

        all_ids = []
        eval_func = self.predict_func
        outputs, predictions, targets = [], [], []
        for ids, idts, idbs, tags in data:
            all_ids += ids
            output, prediction = eval_func(idts, idbs, session)
            outputs.append(output)
            predictions.append(prediction)
            targets.append(tags)

        outputs = np.vstack(outputs)
        predictions = np.vstack(predictions)
        targets = np.vstack(targets).astype(np.int32)  # it was dtype object

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

        all_rankedat10_tags = []
        query_ids = []

        for sample_id, sample_output in zip(eval_samples, outputs):
            q_id = all_ids[sample_id]
            query_ids.append(q_id)
            cols = np.argsort(sample_output)[-10:]
            rankedat10_tags = []
            for col in cols[::-1]:
                label_id = eval_labels[col]
                label_name = tag_names[label_id]
                rankedat10_tags.append(label_name)
            all_rankedat10_tags.append(rankedat10_tags)

        all_Pat5, all_Pat10, all_Rat5, all_Rat10 = \
            ev.Precision(5, True), ev.Precision(10, True), ev.Recall(5, True), ev.Recall(10, True)
        upper_bounds_pat5 = ev.upper_bound(5, True)
        upper_bounds_pat10 = ev.upper_bound(10, True)
        assert len(all_Pat5) == len(all_rankedat10_tags)

        # mat = ev.ConfusionMatrix(5)
        # print_matrix(
        #     mat,
        #     [name for t, name in enumerate(tag_names) if t in eval_labels],
        #     'Confusion:True Tag on x-axis, False Tag on y-axis',
        #     folder)
        # mat = ev.CorrelationMatrix()
        # print_matrix(mat,
        #              [name for t, name in enumerate(tag_names) if t in eval_labels],
        #              'Correlation: True Tag on both axis', folder)

        print 'average all ... ', ev.Precision(5), ev.Precision(10), ev.Recall(5), ev.Recall(10)
        return query_ids, all_rankedat10_tags, list(all_Pat5), list(all_Pat10), list(all_Rat5), list(all_Rat10), \
               upper_bounds_pat5, upper_bounds_pat10


def create_batches(df, ids_corpus, data_type, batch_size, padding_id, perm=None):

    df = df[df['type'] == data_type]
    data_ids = df['id'].values

    if perm is None:  # if no given order (i.e. perm), make a shuffle-random one.
        perm = range(len(data_ids))
        random.shuffle(perm)

    N = len(data_ids)

    cnt = 0
    titles, bodies, tag_labels = [], [], []
    batches = []
    ids = []

    for u in xrange(N):
        i = perm[u]
        q_id = data_ids[i]
        title, body, tag = ids_corpus[str(q_id)]
        cnt += 1
        titles.append(title)
        bodies.append(body)
        tag_labels.append(tag)
        ids.append(q_id)

        if cnt == batch_size or u == N-1:
            titles, bodies, tag_labels = myio.create_one_batch(titles, bodies, tag_labels, padding_id)
            batches.append((ids, titles, bodies, tag_labels))

            titles, bodies, tag_labels = [], [], []
            cnt = 0
            ids = []

    return batches


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
    argparser.add_argument("--results_file", type=str, default="")  # to write in

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

        eval_batches = create_batches(df, ids_corpus, 'test', myqrapi.model.args.batch_size, padding_id)
        print 'TEST evaluation:'
        print '{} batches.'.format(len(eval_batches))
        R = myqrapi.evaluate(eval_batches, label_tags, os.path.join(args.out_dir, 'test') if args.out_dir else None, sess)

        if args.results_file:
            with open(args.results_file, 'w') as f:
                for i in range(len(R[0])):
                    query_id, rankedat10_tags, Pat5, Pat10, Rat5, Rat10, UB5, UB10 = \
                        R[0][i], R[1][i], R[2][i], R[3][i], R[4][i], R[5][i], R[6][i], R[7][i]

                    real_tags = raw_corpus[str(query_id)][2]
                    real_tags = list(set(real_tags) & set(label_tags))
                    real_tags = " ".join([str(x) for x in real_tags])

                    rankedat10_tags = " ".join([str(x) for x in rankedat10_tags])

                    f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                        query_id, real_tags, rankedat10_tags, Pat5, Pat10, Rat5, Rat10, UB5, UB10
                    ))
