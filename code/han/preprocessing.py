import argparse
from qa import myio
import sys
import gzip
from utils import load_embedding_iterator
from nltk.tokenize import sent_tokenize, word_tokenize
from nn import EmbeddingLayer
from collections import Counter
import matplotlib.pyplot as plt
import time
import numpy as np


def read_corpus(path):
    empty_cnt = 0
    raw_corpus = {}
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path) as fin:
        for line in fin:
            id, title, body = line.split("\t")
            if len(title) == 0:
                print id
                empty_cnt += 1
                continue
            title = title.decode("utf8")
            body = body.decode("utf8")
            title = [word_tokenize(sent) for sent in sent_tokenize(title.strip())]
            body = [word_tokenize(sent) for sent in sent_tokenize(body.strip())]
            raw_corpus[id] = (title, body)
    print("{} empty titles ignored.\n".format(empty_cnt))
    return raw_corpus


def create_embedding_layer(n_d, embs=None, unk="<unk>", padding="<padding>", fix_init_embs=True):
    embedding_layer = EmbeddingLayer(
            n_d=n_d,
            # vocab=(w for w,c in cnt.iteritems() if c > cut_off),
            vocab=[unk, padding],
            embs=embs,
            fix_init_embs=fix_init_embs
        )
    return embedding_layer


def map_corpus(raw_corpus, embedding_layer, max_len=100):
    print 'map corpus: for one question we have multiple sentences with words, totalling upt to 100 words...'
    ids_corpus = {}
    for id, pair in raw_corpus.iteritems():

        total_words = 0
        words_ids_per_sent = []

        for sent in pair[0]+pair[1]:
            if total_words >= max_len:
                break
            words_ids_sent = embedding_layer.map_to_ids(sent, filter_oov=True)
            if total_words + len(words_ids_sent) <= max_len:
                words_ids_per_sent.append(words_ids_sent)
                total_words += len(words_ids_sent)
            else:
                words_ids_per_sent.append(words_ids_sent[0:max_len-total_words])
                total_words += len(words_ids_sent[0:max_len-total_words])

        ids_corpus[id] = words_ids_per_sent

    return ids_corpus


def stats(ids_corpus):
    print 'printing statistics on the questions after being left with 100 words in total'
    sent_per_q = Counter()
    words_per_sent = Counter()
    for id, question_words in ids_corpus.iteritems():
        sent_per_q += Counter([len(question_words)])
        for sent in question_words:
            words_per_sent += Counter([len(sent)])

    labels, values = zip(*words_per_sent.items())
    indexes = np.arange(len(labels))
    width = 1
    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels)
    plt.title('words per sentence')
    plt.show()

    labels, values = zip(*sent_per_q.items())
    indexes = np.arange(len(labels))
    width = 1
    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels)
    plt.title('sentences per question')
    plt.show()
    # print 'sent per q ', sent_per_q
    # print 'words per sent ', words_per_sent


def create_eval_batches(ids_corpus, data, padding_id, sent_seq_len=15, word_seq_len=50):
    # always pad right
    lst = []
    for pid, qids, qlabels in data:
        questions = []
        for id in [pid]+qids:
            question_words = ids_corpus[id]
            questions.append(question_words)
        questions = create_one_batch(questions, padding_id, sent_seq_len, word_seq_len)
        lst.append((questions, np.array(qlabels, dtype="int32")))
    return lst


def create_one_batch(questions, padding_id, sent_seq_len=15, word_seq_len=50):
    # always pad right
    new_questions = []
    for question in questions:
        new_sentences = []
        for sentence in question:
            if word_seq_len - len(sentence) < 0:
                new_sentences.append(np.array(sentence[0:word_seq_len]))
            else:
                new_sentences.append(
                    np.pad(sentence, (0, word_seq_len - len(sentence)), 'constant', constant_values=padding_id)
                )
        to_append = sent_seq_len - len(new_sentences)
        if to_append < 0:
            new_sentences = new_sentences[0: sent_seq_len]
        else:
            for _ in range(to_append):
                new_sentences.append(np.zeros(word_seq_len))
        new_questions.append(new_sentences)
    new_questions = np.array(new_questions)
    return new_questions


def main():
    s_time = time.time()
    raw_corpus = read_corpus(args.corpus)
    print 'took ', time.time()-s_time
    print 'raw_corpus example: ', raw_corpus.keys()[0], raw_corpus.values()[0]
    s_time = time.time()

    embedding_layer = create_embedding_layer(
        n_d=args.hidden_dim,
        embs=load_embedding_iterator(args.embeddings) if args.embeddings else None
    )
    print 'took ', time.time()-s_time
    s_time = time.time()

    ids_corpus = map_corpus(raw_corpus, embedding_layer, max_len=args.max_seq_len)
    print 'took ', time.time()-s_time
    print 'ids_corpus example: ', ids_corpus.keys()[0], ids_corpus.values()[0]

    stats(ids_corpus)

    print("vocab size={}, corpus size={}\n".format(
            embedding_layer.n_V,
            len(raw_corpus)
        ))
    padding_id = embedding_layer.vocab_map["<padding>"]

    if args.dev:
        dev = myio.read_annotations(args.dev, K_neg=-1, prune_pos_cnt=-1)
        dev = create_eval_batches(ids_corpus, dev, padding_id)

    if args.test:
        test = myio.read_annotations(args.test, K_neg=-1, prune_pos_cnt=-1)
        test = create_eval_batches(ids_corpus, test, padding_id)

        for t in test:
            print t
            print t[0].shape, t[1].shape
            break

    # todo: add training data

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--corpus", type=str)
    argparser.add_argument("--train", type=str, default="")
    argparser.add_argument("--test", type=str, default="")
    argparser.add_argument("--dev", type=str, default="")

    argparser.add_argument("--embeddings", type=str, default="")
    argparser.add_argument("--hidden_dim", "-d", type=int, default=200)
    argparser.add_argument("--cut_off", type=int, default=1)
    argparser.add_argument("--max_seq_len", type=int, default=100)

    args = argparser.parse_args()
    print args
    print ""
    main()
