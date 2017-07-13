# -*- coding: utf-8 -*-
import os
import scipy
import numpy as np
import gzip
import nltk
import copy
import itertools


this_dir = os.path.dirname(os.path.realpath(__file__))
q_file_name = os.path.join(
	this_dir, 'askubuntu_data', 'texts_raw_fixed.txt.gz')
test_file_name = os.path.join(this_dir, 'askubuntu_data', 'test.txt')
dev_file_name = os.path.join(this_dir, 'askubuntu_data', 'dev.txt')
train_file_name = os.path.join(this_dir, 'askubuntu_data', 'train_random.txt')


def f_open(x):
	if x.endswith(".gz"):
		return gzip.open(x)
	else:
		return open(x)


def read_questions(filename=q_file_name):
	with f_open(filename) as f:
		questions = f.readlines()
		for line in questions:
			q_id, q_title, q_body = line.decode('utf-8').split(u'\t')
			# print "Qid: {}\tQt: {}\tQb: {}".format(q_id, q_title, q_body)
			yield int(q_id), q_title, q_body


def read_rows(ttt='train', filename=train_file_name):
	with f_open(filename) as f:
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


def train_data_triples(title_body=12, **kwargs):

	train_rows = list(read_rows(ttt='train', filename=train_file_name))

	Q = list(read_questions(filename=q_file_name))
	if title_body == 11:
		q_title_body =\
			[u"{} {}".format(var_to_utf(q[1]), var_to_utf(q[2])) for q in Q]
	elif q_title_body == 10:
		q_title_body = [q[1] for q in Q]
	q_indices = [q[0] for q in Q]
	del Q

	tokenized_q, word_to_index = texts_to_words_ids(
		q_title_body,
		kwargs['vocabulary_size'],
		tokenizer=tokenize_sentence
	)
	max_document_length = max([len(xi) for xi in tokenized_q])
	sequence_length = kwargs['sequence_length']
	if max_document_length < sequence_length:
		sequence_length = max_document_length
		print 'sequence length ', sequence_length
		print 'max doc len ', max_document_length
	vectors, word_to_index = words_ids_to_vectors(
		tokenized_q, word_to_index, sequence_length, kwargs['pad']
	)
	del tokenized_q, q_title_body

	questions_idx = questions_index(q_indices, vectors)
	del q_indices, vectors
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
	del train_rows

	# trained_dict, trained_embedding_mat = read_vocabulary()
	# embedding_mat = get_embedding_mat(
	# 	trained_dict, trained_embedding_mat, word_to_index)
	# print '??'

	# return np.array(train_instances), embedding_mat, len(word_to_index.keys())
	return train_instances, word_to_index


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

		# if shuffle:
		# 	shuffle_indices = np.random.permutation(np.arange(data_size))
		# 	shuffled_data = data[shuffle_indices]
		# else:
		# 	shuffled_data = data

		for batch_num in range(num_batches):
			# print 'batch ', batch_num
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield np.array(data[start_index:end_index])


def is_ascii(s):
	return all(ord(c) < 128 for c in s)


def var_to_utf(s):
	if isinstance(s, list):
		return [var_to_utf(i) for i in s]
	if isinstance(s, dict):
		new_dict = dict()
		for key, value in s.items():
			new_dict[var_to_utf(key)] = var_to_utf(copy.deepcopy(value))
		return new_dict
	if isinstance(s, str):
		if is_ascii(s):
			return unicode(s.encode('utf-8'))
		else:
			return unicode(s.decode('utf-8'))
	elif isinstance(s, unicode):
		return s
	elif isinstance(s, int) or isinstance(s, float) or isinstance(s, long):
		return unicode(s)
	elif isinstance(s, tuple):
		return var_to_utf(s[0]), var_to_utf(s[1])
	else:
		print "s: ", s
		print "t: ", type(s)
		raise Exception("unknown type to encode ...")


def read_vocabulary(
	filename='/home/christina/Downloads/glove.6B/glove.6B.200d.txt'
):
	dictionary = {}
	if not os.path.isfile(filename):
		return None, None, None
	embedding_mat = []
	with f_open(filename) as fin:
		lines = -1
		for line in fin:
			lines += 1
			# if lines > 10:
			#     break
			line = line.strip()
			word = var_to_utf(line.split(' ')[0])
			vector = np.array(line.split(' ')[1:])
			# print word, vector
			dictionary[word] = lines
			embedding_mat.append(vector)
		embedding_mat = np.vstack(embedding_mat)
		# print embedding_mat
	return dictionary, embedding_mat


def tokenize_sentence(sent):
	return sent.split(' ')


def texts_to_words_ids(texts, vocabulary_size, tokenizer=tokenize_sentence):
	sentence_start_token = "$START"
	sentence_end_token = "$END"
	unknown_token = "UNK"

	sentences = [text.lower() for text in texts]
	sentences = [
		"%s %s %s" % (
			sentence_start_token, x, sentence_end_token) for x in sentences
	]

	tokenized_sentences = [tokenizer(sent) for sent in sentences]

	# Count the word frequencies
	word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))

	# Get the most common words
	vocab = word_freq.most_common(vocabulary_size - 1)
	index_to_word = [x[0] for x in vocab]
	index_to_word.append(unknown_token)
	word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

	# Replace all words not in our vocabulary with the unknown token
	for i, sent in enumerate(tokenized_sentences):
		tokenized_sentences[i] = [
			w if w in word_to_index else unknown_token for w in sent
		]
	return tokenized_sentences, word_to_index


def words_ids_to_vectors(
		tokenized_sentences, word_to_index, sequence_length, pad='right'):
	padding_token_id = len(word_to_index.keys())  # i.e. len of vocabulary
	word_to_index['$PAD'] = padding_token_id

	X_train = np.asarray(
		[
			# cut upt to sequence_length
			[word_to_index[w] for i, w in enumerate(sent) if i < sequence_length]
			for sent in tokenized_sentences
		]
	)

	if pad == 'right':
		X_train = np.vstack(
			[np.pad(
				x,
				(0, sequence_length - len(x)),
				'constant',
				constant_values=padding_token_id
			 ) for x in X_train]
		)
	else:
		X_train = np.vstack(
			[np.pad(
				x,
				(sequence_length - len(x), 0),
				'constant',
				constant_values=padding_token_id
			 ) for x in X_train]
		)
	return X_train, word_to_index


def get_embedding_mat(
		trained_dictionary, trained_embedding_mat, word_to_index):
	if trained_embedding_mat is None:
		return None
	vocabulary_size = len(word_to_index.keys())
	embedding_size = trained_embedding_mat.shape[1]
	embedding_mat = np.random.uniform(
		low=-1.0, high=1.0, size=(vocabulary_size, embedding_size))

	for word, word_id in word_to_index.iteritems():
		if word in trained_dictionary:
			idx = trained_dictionary[word]
			# print 'word: {} with idx {} has idx {} in pretrained '.format(
			#     word, word_id, idx)
			embedding_mat[word_id] = trained_embedding_mat[idx]
	return embedding_mat


if __name__ == '__main__':

	X, w2i = train_data_triples(
	    title_body=11,
	    vocabulary_size=200000,
	    sequence_length=100,
	    pad='right'
	)

	trained_dict, trained_embedding_mat = read_vocabulary()
	embedding_mat = get_embedding_mat(
		trained_dict, trained_embedding_mat, w2i)
	del trained_dict, trained_embedding_mat, w2i

	batches = batch_iter(X, 64, 3)
	for b in batches:
		print b.shape
		print b[0]
		exit()
