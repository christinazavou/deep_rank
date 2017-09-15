import argparse
import sys
from utils import load_embedding_iterator
import time
from tags_prediction.myio import read_corpus
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def jaccard_similarity(a, b):
    a = (a > 0).astype(np.float32)  # ok with sparse as well
    b = (b > 0).astype(np.float32)
    if b.ndim > 1:
        a = a.toarray()
        b = b.toarray()
        intersect = (a * b).sum(axis=1)
        result = intersect / np.tile(b.shape[1], b.shape[0])
    else:
        intersect = (a * b).sum()  # ok with sparse as well
        result = intersect / len(a)
    return result


class Categorization(object):

    def __init__(self, items):
        # items = ['<5', '<8', '<14', '>=14']
        # items = ["what", "when", "where", "why", "how", "who"]
        # items = [0.05, 1, 0.1, 0.2]
        # items = [0.1, 0.2, 0.3, 0.4, 0.5]

        # map(hash, list(itertools.combinations(['<5', '<8', '<14', '>14'], 2)))
        self.items = items
        if isinstance(items[0], str):
            l = list(itertools.combinations(items, 2)) + [(i, i) for i in items]
            self.categories = {c: i for i, c in enumerate(l)}
            self.size = len(self.categories)
        else:
            self.size = len(items) + 1
        print 'done categorizer of size ', self.size

    def transform(self, c1, c2):
        # return hash((c1, c2))
        vector = np.zeros(self.size)
        # return self.categories[(c1, c2)]
        try:
            vector[self.categories[(c1, c2)]] = 1
        except:
            vector[self.categories[(c2, c1)]] = 1
        return vector

    def transform_list(self, c1, c2):
        vector = np.zeros(self.size)
        for i1 in c1:
            for i2 in c2:
                try:
                    vector[self.categories[(i1, i2)]] = 1
                except:
                    vector[self.categories[(i2, i1)]] = 1
        return vector

    def transform_item(self, c):
        vector = np.zeros(self.size)
        for i, item in enumerate(self.items):
            if c < item:
                vector[i] = 1
                return vector
        vector[-1] = 1
        return vector


def corpus_grams(raw_corpus, max_seq_len=1000, n_gram=3):
    s_time = time.time()

    corpus = [" ".join(title) + " " + " ".join(body[0:max_seq_len]) for qid, (title, body, tags) in raw_corpus.iteritems()]

    tf_idf_vec = TfidfVectorizer(ngram_range=(1, n_gram), max_df=0.75, min_df=5, )
    # tf_vec = TfidfVectorizer(ngram_range=(1, n_gram), max_df=0.75, min_df=5, use_idf=False)
    corpus_vectorized = tf_idf_vec.fit_transform(corpus)

    pos_tagged_corpus = []
    for qid, (title, body, tags) in raw_corpus.iteritems():
        title_pos_tags = [w[1] for w in nltk.pos_tag(title)]
        body_pos_tags = [w[1] for w in nltk.pos_tag(body[0:max_seq_len])]
        pos_tagged_corpus.append(" ".join(title_pos_tags) + " " + " ".join(body_pos_tags))

    tf_idf_pos_tag_vec = TfidfVectorizer(ngram_range=(1, n_gram), max_df=0.75, min_df=5, )
    # tf_pos_tag_vec = TfidfVectorizer(ngram_range=(1, n_gram), max_df=0.75, min_df=5, use_idf=False)
    pos_tagged_corpus_vectorized = tf_idf_pos_tag_vec.fit_transform(pos_tagged_corpus)

    del corpus, pos_tagged_corpus

    corpus_dict = {}
    corpus_postags_dict = {}
    idx = 0
    for qid, (title, body, tags) in raw_corpus.iteritems():
        corpus_dict[qid] = corpus_vectorized[idx]
        corpus_postags_dict[qid] = pos_tagged_corpus_vectorized[idx]
        idx += 1

    print 'took ', (time.time()-s_time)//60, ' for the vectorization of text.'  # 19 mins
    return corpus_dict, corpus_postags_dict


def questions_indicators(raw_corpus, max_seq_len=10000):
    s_time = time.time()
    qis = ["what", "when", "where", "why", "how", "who"]
    qis_corpus = {}
    for qid, (title, body, tags) in raw_corpus.iteritems():
        cur_qis = []
        for i, qi in enumerate(qis):
            if qi in title or qi in body[0:max_seq_len]:
                cur_qis.append(qi)
        qis_corpus[qid] = cur_qis
    print 'took ', time.time()-s_time, ' for questions indicators.'
    return qis_corpus


def sub_questions(raw_corpus, max_seq_len=10000):
    s_time = time.time()
    # after looking at the histogram decided categories
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    num_sentences_corpus = {}
    # nums = [4, 7, 10, 13]
    for qid, (title, body, tags) in raw_corpus.iteritems():
        num = len(tokenizer.tokenize((" ".join(title)+" "+" ".join(body[0:max_seq_len])).decode('utf8')))
        if num < 5:
            num_sentences_corpus[qid] = "<5"
        elif num < 8:
            num_sentences_corpus[qid] = "<8"
        elif num < 14:
            num_sentences_corpus[qid] = "<14"
        else:
            num_sentences_corpus[qid] = ">=14"
    print 'took ', time.time() - s_time, ' for sub_questions.'
    return num_sentences_corpus


def main(test=False):

    """--------------------------------------------------------------------------------------------------------------"""
    raw_corpus = read_corpus(args.corpus, with_tags=True, test=-1 if not test else 10000)

    corpus_dict, corpus_postags_dict = corpus_grams(
        raw_corpus, max_seq_len=10000 if not test else 100, n_gram=2)

    num_sentences_corpus = sub_questions(raw_corpus, max_seq_len=10000 if not test else 100)

    qis_corpus = questions_indicators(raw_corpus, max_seq_len=10000 if not test else 100)
    """--------------------------------------------------------------------------------------------------------------"""

    """--------------------------------------------------------------------------------------------------------------"""
    sent_num_categorizer = Categorization(['<5', '<8', '<14', '>=14'])
    q_identifiers_categorizer = Categorization(["what", "when", "where", "why", "how", "who"])
    cosine_vectorizer = Categorization([0.05, 1, 0.1, 0.2])
    tag_cosine_vectorizer = Categorization([0.1, 0.2, 0.3, 0.4, 0.5])
    bins = [
        0,
        cosine_vectorizer.size, cosine_vectorizer.size + tag_cosine_vectorizer.size,
        cosine_vectorizer.size + tag_cosine_vectorizer.size + sent_num_categorizer.size,
        cosine_vectorizer.size + tag_cosine_vectorizer.size + sent_num_categorizer.size + q_identifiers_categorizer.size
    ]
    print 'bins ', bins
    """--------------------------------------------------------------------------------------------------------------"""

    def make_train_file(in_file, out_file):

        test_cosines = []
        test_tags_cosines = []

        with open(in_file, 'r') as f_in:
            lines = f_in.readlines()
            with open(out_file, 'w') as f_out:
                for line_id, line in enumerate(lines):

                    if line_id % 500 == 0:
                        print 'done line ', line_id
                    if line_id == 10000 and test:
                        break

                    qid, qsim, qdif = line.split("\t")

                    qsim = qsim.strip().split()
                    qdif = qdif.strip().split()

                    q_sim_vectors = []
                    q_dif_vectors = []

                    for pid in qsim:

                        if (pid not in corpus_dict or qid not in corpus_dict) and test:
                            continue

                        vector = np.zeros(
                            cosine_vectorizer.size + tag_cosine_vectorizer.size +
                            sent_num_categorizer.size + q_identifiers_categorizer.size
                        )

                        cos = cosine_similarity(corpus_dict[qid], corpus_dict[pid])[0][0]
                        tag_cos = cosine_similarity(corpus_postags_dict[qid], corpus_postags_dict[pid])[0][0]

                        vector[bins[0]:bins[1]] = cosine_vectorizer.transform_item(cos)
                        vector[bins[1]:bins[2]] = tag_cosine_vectorizer.transform_item(tag_cos)

                        vector[bins[2]:bins[3]] = sent_num_categorizer.transform(
                            num_sentences_corpus[qid], num_sentences_corpus[pid]
                        )
                        vector[bins[3]:bins[4]] = q_identifiers_categorizer.transform_list(
                            qis_corpus[qid], qis_corpus[pid]
                        )

                        q_sim_vectors.append(" ".join(str(x) for x in vector))

                        test_cosines.append(cos)
                        test_tags_cosines.append(tag_cos)

                    for pid in qdif:

                        if (pid not in corpus_dict or qid not in corpus_dict) and test:
                            continue

                        vector = np.zeros(
                            cosine_vectorizer.size + tag_cosine_vectorizer.size +
                            sent_num_categorizer.size + q_identifiers_categorizer.size
                        )

                        cos = cosine_similarity(corpus_dict[qid], corpus_dict[pid])[0][0]
                        tag_cos = cosine_similarity(corpus_postags_dict[qid], corpus_postags_dict[pid])[0][0]

                        vector[bins[0]:bins[1]] = cosine_vectorizer.transform_item(cos)
                        vector[bins[1]:bins[2]] = tag_cosine_vectorizer.transform_item(tag_cos)

                        vector[bins[2]:bins[3]] = sent_num_categorizer.transform(
                            num_sentences_corpus[qid], num_sentences_corpus[pid]
                        )
                        vector[bins[3]:bins[4]] = q_identifiers_categorizer.transform_list(
                            qis_corpus[qid], qis_corpus[pid]
                        )

                        q_dif_vectors.append(" ".join(str(x) for x in vector))

                        test_cosines.append(cos)
                        test_tags_cosines.append(tag_cos)

                    f_out.write("{}\t{}\t{}\t{}\t{}\n".format(
                        qid, " ".join(qsim), " ".join(qdif), ", ".join(q_sim_vectors), ", ".join(q_dif_vectors)))

        plt.hist(test_cosines, bins=np.linspace(0, 0.6, 20))
        plt.show()
        plt.hist(test_tags_cosines, bins=np.linspace(0, 0.8, 20))
        plt.show()

    def make_eval_file(in_file, out_file):
        with open(in_file, 'r') as f_in:
            lines = f_in.readlines()
            with open(out_file, 'w') as f_out:
                for line in lines:
                    qid, qsim, qcand, qscores = line.split("\t")

                    qsim = qsim.strip().split()
                    qcand = qcand.strip().split()

                    qvectors = []

                    for pid in qcand:

                        if (pid not in corpus_dict or qid not in corpus_dict) and test:
                            continue

                        vector = np.zeros(
                            cosine_vectorizer.size + tag_cosine_vectorizer.size +
                            sent_num_categorizer.size + q_identifiers_categorizer.size
                        )

                        cos = cosine_similarity(corpus_dict[qid], corpus_dict[pid])[0][0]
                        tag_cos = cosine_similarity(corpus_postags_dict[qid], corpus_postags_dict[pid])[0][0]

                        vector[bins[0]:bins[1]] = cosine_vectorizer.transform_item(cos)
                        vector[bins[1]:bins[2]] = tag_cosine_vectorizer.transform_item(tag_cos)

                        vector[bins[2]:bins[3]] = sent_num_categorizer.transform(
                            num_sentences_corpus[qid], num_sentences_corpus[pid]
                        )
                        vector[bins[3]:bins[4]] = q_identifiers_categorizer.transform_list(
                            qis_corpus[qid], qis_corpus[pid]
                        )

                        qvectors.append(" ".join(str(x) for x in vector))

                    f_out.write("{}\t{}\t{}\t{}\t{}\n".format(
                        qid, " ".join(qsim), " ".join(qcand), qscores.strip(), ", ".join(qvectors)))

    s_time = time.time()
    make_train_file(args.train, args.new_train)
    print 'train file done. after ', (time.time() - s_time)//60  # 28 mins
    make_eval_file(args.dev, args.new_dev)
    print 'dev file done.'
    make_eval_file(args.test, args.new_test)
    print 'test file done.'


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--corpus", type=str)
    argparser.add_argument("--train", type=str, default="")
    argparser.add_argument("--new_train", type=str, default="")
    argparser.add_argument("--test", type=str, default="")
    argparser.add_argument("--new_test", type=str, default="")
    argparser.add_argument("--dev", type=str, default="")
    argparser.add_argument("--new_dev", type=str, default="")

    argparser.add_argument("--embeddings", type=str, default="")
    argparser.add_argument("--max_seq_len", type=int, default=100)

    args = argparser.parse_args()
    print args
    print ""
    main(False)
