import os
import scipy
import numpy as np
from deep_rank.texts_parse import parse


this_dir = os.path.dirname(os.path.realpath(__file__))
q_file_name = os.path.join(this_dir, 'askubuntu_data', 'texts_raw_fixed.txt')
test_file_name = os.path.join(this_dir, 'askubuntu_data', 'test.txt')
dev_file_name = os.path.join(this_dir, 'askubuntu_data', 'dev.txt')
train_file_name = os.path.join(this_dir, 'askubuntu_data', 'train_random.txt')


def read_questions(filename=q_file_name):
	with open(filename) as f:
		questions = f.readlines()
		for line in questions:
			q_id, q_title, q_body = line.decode('utf-8').split(u'\t')
			# print "Qid: {}\tQt: {}\tQb: {}".format(q_id, q_title, q_body)
			yield int(q_id), q_title, q_body


def read_rows(ttt='train', filename=train_file_name):
	with open(filename) as f:
		questions = f.readlines()

		if ttt == 'train':
			for line in questions:
				q_id, q_ids_similar, q_ids_dissimilar = \
					line.decode('utf-8').split(u'\t')
				# print "Qid: {}\tQsim: {}\tQdis: {}".format(
				# 	q_id, q_ids_similar, q_ids_dissimilar
				# )
				q_ids_similar = q_ids_similar.split(' ')
				q_ids_dissimilar = q_ids_dissimilar.split(' ')

				q_ids_similar = strlist_to_intlist(q_ids_similar)
				q_ids_dissimilar = strlist_to_intlist(q_ids_dissimilar)

				yield int(q_id), q_ids_similar, q_ids_dissimilar
		else:
			for line in questions:
				q_id, q_ids_similar, q_ids_candidates, q_bm25_candidates = \
					line.decode('utf-8').split(u'\t')
				# print "Qid: {}\tQsim: {}\tQcand: {}\tQbm25: {}".format(
				# 	q_id, q_ids_similar, q_ids_candidates, q_bm25_candidates
				# )
				q_ids_similar = q_ids_similar.split(' ')
				q_ids_candidates = q_ids_candidates.split(' ')
				q_bm25_candidates = q_bm25_candidates.split(' ')

				q_ids_similar = strlist_to_intlist(q_ids_similar)
				q_ids_candidates = strlist_to_intlist(q_ids_candidates)
				q_bm25_candidates = [float(c) for c in q_bm25_candidates]

				yield int(q_id), q_ids_similar, q_ids_candidates, q_bm25_candidates


def strlist_to_intlist(strlist):
	result = []
	try:
		result = [int(item) for item in strlist]
	except:
		# print('except: {}\n'.format(strlist))
		pass
	return result


def questions_index(questions_ids, questions_vectors):
	q_idx = {}
	for i in range(len(questions_ids)):
		q_id = questions_ids[i]
		tf_vector = questions_vectors[i]
		if isinstance(tf_vector, scipy.sparse.csr_matrix):
			q_idx[q_id] = tf_vector.toarray()
		else:
			q_idx[q_id] = tf_vector
	return q_idx


def train_data_triples(**kwargs):

	train_rows = list(read_rows(ttt='train', filename=train_file_name))

	Q = list(read_questions(filename=q_file_name))

	vectors, num_words = parse([q[1] for q in Q], **kwargs)

	questions_idx = questions_index([q[0] for q in Q], vectors)

	train_instances = []
	for q_id, q_ids_similar, q_ids_dissimilar in train_rows:
		for q_id_sim in q_ids_similar:
			for q_id_dis in q_ids_dissimilar:
				train_instances += [
					(questions_idx[q_id],
					 questions_idx[q_id_sim],
					 questions_idx[q_id_dis])
				]
				# returns triples of (Q, Q+, Q-)
	return np.array(train_instances), num_words


def split_data(x, dev_sample_percentage):
	# Randomly shuffle data
	np.random.seed(10)
	shuffle_indices = np.random.permutation(np.arange(len(x)))
	x_shuffled = x[shuffle_indices]

	# Split train/test set
	dev_sample_index = -1 * int(dev_sample_percentage * float(len(x)))
	x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
	print("Train/Dev split: {:d}/{:d}".format(len(x_train), len(x_dev)))

	return x_train, x_dev


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    # data = np.array(data)
    data_size = len(data)
    num_batches = int((len(data) - 1) / batch_size) + 1  # per epoch
    print ' will have a total of ', num_batches * num_epochs, ' batches '

    for epoch in range(num_epochs):
        # print 'epoch ', epoch

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches):
            # print 'batch ', batch_num
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
