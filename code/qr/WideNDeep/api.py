# import argparse
# import os
# import sys
#
# import numpy as np
# import scipy.sparse
# import tensorflow as tf
# from sklearn.metrics.pairwise import cosine_similarity
#
# import qa.myio
# from qa.WideNDeep.main_model import Model
# from qa.evaluation import Evaluation
# from qa.myio import say
# from utils import load_embedding_iterator
#
#
# class QRAPI:
#
#     def __init__(self, model_path, corpus_path, emb_path, session):
#         raw_corpus = qa.myio.read_corpus(corpus_path)
#         embedding_layer = qa.myio.create_embedding_layer(
#                     raw_corpus,
#                     n_d = 10,
#                     cut_off = 1,
#                     embs = load_embedding_iterator(emb_path)
#                 )
#         # weights = myio.create_idf_weights(corpus_path, embedding_layer)
#         say("vocab size={}, corpus size={}\n".format(
#                 embedding_layer.n_V,
#                 len(raw_corpus)
#             ))
#
#         # model = Model(args=None, embedding_layer=embedding_layer,
#         #             weights=weights)
#         model = Model(args=None, embedding_layer=embedding_layer)
#
#         model.load_n_set_model(model_path, session)
#         say("model initialized\n")
#
#         self.model = model
#
#         def score_func(titles, bodies, features, cur_sess):
#             extended_features = np.zeros((21, features.shape[1]))
#             extended_features[0:-1] = features  # we add a dummy instance at the end
#             _scores = cur_sess.run(
#                 self.model.scores,
#                 feed_dict={
#                     self.model.titles_words_ids_placeholder: titles.T,  # IT IS TRANSPOSE ;)
#                     self.model.bodies_words_ids_placeholder: bodies.T,  # IT IS TRANSPOSE ;)
#                     self.model.dropout_prob: 0.,
#                     self.model.features_placeholder: np.reshape(extended_features, (1, 21, features.shape[1]))  # batch size 1
#
#                 }
#             )
#             return _scores
#
#         self.score_func = score_func
#         say("scoring function compiled\n")
#
#     def evaluate(self, data, session):
#
#         eval_func = self.score_func
#         res_ranked_labels = []
#         res_ranked_ids = []
#         res_ranked_scores = []
#         query_ids = []
#         all_MAP, all_MRR, all_Pat1, all_Pat5 = [], [], [], []
#         for idts, idbs, labels, features, pid, qids in data:
#             scores = eval_func(idts, idbs, features, session)
#             assert len(scores) == len(labels)
#             ranks = (-scores).argsort()
#             ranked_scores = np.array(scores)[ranks]
#             ranked_labels = labels[ranks]
#             ranked_ids = np.array(qids)[ranks]
#             query_ids.append(pid)
#             res_ranked_labels.append(ranked_labels)
#             res_ranked_ids.append(ranked_ids)
#             res_ranked_scores.append(ranked_scores)
#             this_ev = Evaluation([ranked_labels])
#             all_MAP.append(this_ev.MAP())
#             all_MRR.append(this_ev.MRR())
#             all_Pat1.append(this_ev.Precision(1))
#             all_Pat5.append(this_ev.Precision(5))
#
#         print 'average all ... ', sum(all_MAP)/len(all_MAP), sum(all_MRR)/len(all_MRR), sum(all_Pat1)/len(all_Pat1), sum(all_Pat5)/len(all_Pat5)
#         return all_MAP, all_MRR, all_Pat1, all_Pat5, res_ranked_labels, res_ranked_ids, query_ids, res_ranked_scores
#
#
# def create_feature_array(f1_mat, f2_mat):
#     features = np.zeros((f1_mat.shape[0]-1, 2))
#     features[:, 0] = cosine_similarity(f1_mat[0], f1_mat[1:])[0]
#     features[:, 1] = cosine_similarity(f2_mat[0], f2_mat[1:])[0]
#     # todo: jaccard of non vectors
#     return features
#
#
# def create_eval_batches(ids_corpus, data, padding_id, f1_vectors, f2_vectors, pad_left):
#     # returns actually a list of tuples (titles, bodies, qlabels) - each tuple defined for one eval instance
#     # i.e. ([21x100], [21x100], [21x1]) if 10 pos and 20 neg
#     # and so tuples can have different title/body shapes
#     lst = []
#
#     ids = ids_corpus.keys()
#
#     for i, (pid, qids, qlabels) in enumerate(data):
#
#         if i % 100 == 0:
#             print 'i is ', i
#         if i == 10:  # TEST
#             break
#
#         titles = []
#         bodies = []
#
#         vec_ids = []
#
#         for id in [pid]+qids:
#             t, b = ids_corpus[id]
#             titles.append(t)
#             bodies.append(b)
#
#             vec_ids.append(ids.index(id))
#
#         titles, bodies = qa.myio.create_one_batch(titles, bodies, padding_id, pad_left)
#         # print 't ', titles.shape, ' b ', bodies.shape, ' vec_ids ', len(vec_ids)
#
#         features = create_feature_array(f1_vectors[vec_ids], f2_vectors[vec_ids])
#
#         lst.append((titles, bodies, np.array(qlabels, dtype="int32"), features, pid, qids))
#     return lst
#
#
# def make_features_dicts():
#     if os.path.isfile('/home/christina/Documents/Thesis/deep_rank/n_gram_vectors.npy.npz') and\
#             os.path.isfile('/home/christina/Documents/Thesis/deep_rank/pos_tag_n_gram_vectors.npy.npz'):
#         n_gram_vec = scipy.sparse.load_npz('/home/christina/Documents/Thesis/deep_rank/n_gram_vectors.npy.npz')
#         n_gram_pos_tag_vec = scipy.sparse.load_npz('/home/christina/Documents/Thesis/deep_rank/pos_tag_n_gram_vectors.npy.npz')
#         return n_gram_vec, n_gram_pos_tag_vec
#     else:
#         raise Exception('OPA')
#
#
# if __name__ == '__main__':
#     argparser = argparse.ArgumentParser(sys.argv[0])
#     argparser.add_argument("--model_path", type=str)
#     argparser.add_argument("--corpus_path", type=str, default="")
#     argparser.add_argument("--emb_path", type=str, default="")
#     argparser.add_argument("--dev", type=str, default="")
#     argparser.add_argument("--results_file", type=str, default="")
#     args = argparser.parse_args()
#     print '\n', args, '\n'
#
#     with tf.Session() as sess:
#
#         myqrapi = QRAPI(args.model_path, args.corpus_path, args.emb_path, sess)
#
#         raw_corpus = qa.myio.read_corpus(args.corpus_path)
#         embedding_layer = myqrapi.model.embedding_layer
#         ids_corpus = qa.myio.map_corpus(raw_corpus, embedding_layer, max_len=100)
#
#         n_gram_vectors, pos_tag_n_gram_vectors = make_features_dicts()
#
#         dev = qa.myio.read_annotations(args.dev, K_neg=-1, prune_pos_cnt=-1)
#         dev = create_eval_batches(
#             ids_corpus, dev, myqrapi.model.padding_id, n_gram_vectors, pos_tag_n_gram_vectors,
#             pad_left=not myqrapi.model.args.average
#         )
#
#         devmap, devmrr, devpat1, devpat5, rank_labels, rank_ids, qids, rank_scores = myqrapi.evaluate(dev, sess)
#
#         with open(args.results_file, 'w') as f:
#             for i, (_, _, labels, features, pid, qids) in enumerate(dev):
#                 print_qids_similar = [x for x, l in zip(qids, labels) if l == 1]
#                 print_qids_similar = " ".join([str(x) for x in print_qids_similar])
#
#                 print_qids_candidates = " ".join([str(x) for x in rank_ids[i]])
#
#                 print_ranked_scores = " ".join([str(x) for x in rank_scores[i]])
#
#                 print_ranked_labels = " ".join([str(x) for x in rank_labels[i]])
#
#                 f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
#                     pid, print_qids_similar, print_qids_candidates,
#                     print_ranked_scores,
#                     print_ranked_labels,
#                     round(devmap[i],4), round(devmrr[i],4), round(devpat1[i],4), round(devpat5[i],4)
#                 ))