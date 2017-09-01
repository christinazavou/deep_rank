from bs4 import BeautifulSoup
import os
import nltk
from multiprocessing.queues import SimpleQueue
from multiprocessing import Process, freeze_support
import gzip
import copy
from utils import read_questions_with_tags, read_df, store_df, read_eval_rows
from collections import OrderedDict
import pandas as pd
from splitter import get_eval_ids, make_data_frame_for_tag_training


NUM_PROCESSES = 8
MAX_LEN = -1


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
        print "s: ",  s
        print "t: ", type(s)
        raise Exception("unknown type to encode ...")


def parse_question(post):
    """
    Parses a post retrieved from the xml file of AskUbuntu2014,
    returning its id, title, body and accepted answer id
    """
    soup = BeautifulSoup(post, "html.parser")
    if not soup.row:
        print 'Warning : {}, {}'.format(post, soup)
    id = int(soup.row["id"])
    try:
        title = soup.row["title"]
    except:
        print "\rNo title found for post {}; skip.\n".format(id)
        return None, None, None, None
    try:
        body = soup.row["body"]
    except:
        print "\rNo body found for post {}; skip.\n".format(id)
        return None, None, None, None
    try:
        tags = soup.row["tags"]
    except:
        tags = ''
        print 'No tags found for post {} '.format(id)
    body_soup = BeautifulSoup(body, "lxml")

    # remove "Possible Duplicate" section
    blk = body_soup.blockquote
    if blk and blk.text.strip().startswith("Possible Duplicate:"):
        blk.decompose()
    body_cleaned = body_soup.text
    assert "Possible Duplicate:" not in body_cleaned

    title_words = [w.lower() for w in nltk.word_tokenize(title)]
    body_words = [w.lower() for w in nltk.word_tokenize(body_cleaned)]

    title_text = " ".join(title_words)
    body_text = " ".join(body_words)
    assert "\n" not in body_text

    return id, title_text, body_text, tags


def intermediate_parse_worker(queue_in, queue_out):
    """
    A function called by each parallel process to parse the posts of the xml AskUbuntu2014 file.
    Each post is passed here through queue_in, and results (from parse_question or parse_answer) are pushed in queue_out
    """
    while True:

        post = queue_in.get()

        if post is None:
            break
        if "PostTypeId=\"1\"" in post:
            # print 'parsing question'
            results = parse_question(post)
            if results[0] is not None:
                queue_out.put(results)

    queue_out.put(None)


def collector(queue_out, filename):
    """
    Collects all parsed posts that were pushed in queue_out and saves them in a pandas dataframe
    """
    data = []
    cnt = 0
    processed = 0
    while cnt < NUM_PROCESSES:
        item = queue_out.get()
        if item is None:
            cnt += 1  # each Process puts None when finished, so we get results of all processes.
        else:
            data.append(item)  # item is (id, title, body, tags)
            processed += 1
            if processed % 1000 == 0:
                print "\r{}".format(processed)

    data = sorted(data, key=lambda x: x[0])  # sorted by id (all questions and answers together)
    store_to_file(data, filename)


def store_to_file(data, filename):
    """
    Stores a dataframe with columns: 'id', 'title', 'body', 'tags'
    """
    N = len(data)

    with open(filename, 'w') as f:

        for row in xrange(N):
            if row % 1000 == 0:
                print 'row ', row, ' in df'
            assert row == 0 or data[row][0] > data[row-1][0]
            instance = data[row]

            str_tags = instance[3]
            str_tags = str_tags.lstrip(u'<').rstrip(u'>').split(u'><')
            tags_text = u", ".join(str_tags)

            f.write(
                u'{}\t{}\t{}\t{}\n'.format(
                    var_to_utf(instance[0]), var_to_utf(instance[1]), var_to_utf(instance[2]), tags_text
                ).encode('utf8')
            )


def f_open(x):
    if x.endswith(".gz"):
        return gzip.open(x)
    else:
        return open(x)


class UbuntuFileIterator(object):

    def __init__(self, file_name):
        self.lines = 0
        self.file_name = file_name
        with f_open(file_name) as f:
            for line in f.readlines():
                self.lines += 1

    def __iter__(self):
        with f_open(self.file_name) as f:
            for line in f.readlines():
                yield line
            print 'all lines read', self.lines

    def __len__(self):
        return self.lines


def main(infile, outfile):
    test = False

    freeze_support()

    if not os.path.isfile(outfile):

        queue_in_ = SimpleQueue()
        queue_out_ = SimpleQueue()
        processes = []

        for i in xrange(NUM_PROCESSES):
            p = Process(target=intermediate_parse_worker, args=(queue_in_, queue_out_))
            p.start()
            processes.append(p)

        collector_process = Process(target=collector, args=(queue_out_, outfile))
        collector_process.start()

        print "\n"
        print "Reading raw xml file: ", infile
        cnt_ = 0

        fopen = lambda x: gzip.open(x) if x.endswith(".gz") else open(x)

        with fopen(infile) as fin:
            lines = -1
            for line in fin:
                lines += 1
                if test and lines > 100:
                    break
                line = line.strip()
                if line.startswith("<row Id=\""):
                    queue_in_.put(line)
                cnt_ += 1
                if cnt_ % 1000 == 0:
                    print "\r{} lines processed".format(cnt_)
        print "\nDone.\n"

        for i in xrange(NUM_PROCESSES):
            queue_in_.put(None)

        for p in processes:
            p.join()
        collector_process.join()


def make_save_dataframe(infile, outfile, chunk_size=5000):

    if os.path.isfile(outfile):
        return

    Q = list(read_questions_with_tags(infile))
    print 'num of questions ', len(Q)
    tags_set = set()
    for q in Q:
        tags_set.update(set(q[3]))
    columns = ['id', 'title', 'body', 'tags'] + [tag for tag in tags_set]

    def label(column_name, instance_tags):
        if column_name in instance_tags:
            return 1
        return 0

    chunk = 0
    ids, titles, bodies, tags = [], [], [], []
    tags_separated = [[] for _ in columns[4:]]
    init_idx = 0

    for row, q in enumerate(Q):

        ids.append(q[0])
        titles.append(q[1])
        bodies.append(q[2])
        tags.append(u', '.join(q[3]))

        for c, t in enumerate(columns[4:]):
            tags_separated[c].append(label(t, q[3]))

        if len(ids) == chunk_size or row == len(Q) - 1:

            data = OrderedDict()
            for col_idx, col_data in zip(columns, [ids, titles, bodies, tags] + tags_separated):
                data[col_idx] = col_data
            new_chunk = pd.DataFrame(data, index=range(init_idx, row + 1))
            new_chunk = new_chunk.fillna(u'')
            header = True if chunk == 0 else False
            store_df(new_chunk, outfile, True, header, 'a')
            print 'chunk up to {} row put.'.format(row)
            chunk += 1
            ids, titles, bodies, tags = [], [], [], []
            tags_separated = [[] for _ in columns[4:]]
            init_idx = row + 1


if __name__ == '__main__':
    main(
        '/home/christina/Documents/Thesis/data/askubuntu/askubuntu_2014_posts.xml',
        '/home/christina/Documents/Thesis/data/askubuntu/texts_raw_with_tags.txt'
    )

    make_save_dataframe(
        '/home/christina/Documents/Thesis/data/askubuntu/texts_raw_with_tags.txt',
        '/home/christina/Documents/Thesis/data/askubuntu/data_frame_corpus.csv'
    )

    df = read_df('/home/christina/Documents/Thesis/data/askubuntu/data_frame_corpus.csv')

    if 'type' not in list(df):
        E = read_eval_rows('/home/christina/Documents/Thesis/data/askubuntu/test.txt')
        test_ids = get_eval_ids(E)
        E = read_eval_rows('/home/christina/Documents/Thesis/data/askubuntu/dev.txt')
        dev_ids = get_eval_ids(E)

        df = make_data_frame_for_tag_training(df, list(test_ids), list(dev_ids))
        store_df(df, '/home/christina/Documents/Thesis/data/askubuntu/data_frame_corpus.csv')

