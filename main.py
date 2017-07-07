# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from deep_rank.load_data_utils import train_data_triples,\
	split_data, batch_iter
from deep_rank.cnn_config import FLAGS


if __name__ == '__main__':

	X, num_words = train_data_triples(
		vocabulary_size=FLAGS.vocabulary_size,
		sequence_length=FLAGS.sequence_length,
		pad=FLAGS.pad)

	x_train, x_dev = split_data(X, FLAGS.dev_sample_percentage)

 	batches = batch_iter(
 		x_train,
 		FLAGS.batch_size,
 		FLAGS.num_epochs,
 		shuffle=True
 	)
 	for batch in batches:
		print batch
		exit()

	# Q = list(read_questions())
	# E = list(read_eval_rows())
	# T = list(read_train_rows())
	# print len(Q), len(E), len(T)

	# all_q_id = [e[0] for e in E]
	# all_q_ids_similar = [e[1] for e in E]
	# all_q_ids_candidates = [e[2] for e in E]
	# all_q_bm25_candidates = [e[3] for e in E]

	# print 'MAP ', evaluate_recommendations(
	# 	all_q_ids_candidates, all_q_ids_similar, metric='MAP'
	# )
	# print 'MRR ', evaluate_recommendations(
	# 	all_q_ids_candidates, all_q_ids_similar, metric='MRR'
	# )
	# print 'P@1 ', evaluate_recommendations(
	# 	all_q_ids_candidates, all_q_ids_similar, metric='P@1'
	# )
	# print 'P@5 ', evaluate_recommendations(
	# 	all_q_ids_candidates, all_q_ids_similar, metric='P@5'
	# )
