import matplotlib.pyplot as plt
import operator
from utils import read_df
import pickle
import os
import numpy as np


def frqplot(tags=['tag' for _ in range(50)], freq=range(50), title='tag freq. in train'):

    ind = np.arange(len(freq))  # the x locations for the groups
    width = 0.1       # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(ind, freq, width, color='r')
    # add some text for labels, title and axes ticks
    ax.set_title(title)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(tags, rotation=50)
    plt.show()


def hist_num_of_tags(data_frame, type_name=None):
    if type_name:
        data_frame = data_frame[data_frame['type'] == type_name]

    num_of_tags = [len(row_['tags'].split(', ')) for _, row_ in data_frame.iterrows()]

    plt.hist(num_of_tags)
    plt.title('tags per question in {} corpus'.format(type_name if type_name else 'whole'))
    plt.xlabel('amount of tags')
    plt.show()

    print 'num of questions with 0 tags ', num_of_tags.count(0)
    print 'avg, min, max number of tags per question ', sum(num_of_tags)/(len(data_frame)+0.), min(num_of_tags), max(num_of_tags)
    print 'std : ', np.std(np.array(num_of_tags))


def tags_not_in_questions(data_frame, type_name=None):
    if type_name:
        data_frame = data_frame[data_frame['type'] == type_name]
    tags_not_in_question = []
    tags_not_in = set()
    for _, row_ in data_frame.iterrows():
        ans = 0
        try:
            for t in row_['tags'].split(u", "):
                if t not in row_['title'] and t not in row_['body']:
                    ans += 1
                    tags_not_in.update({t})
        except:
            print 'in except: ', row_['tags'], type(row_['tags'])
        tags_not_in_question.append(ans)

    print 'some tags not in questions happens in ', len(data_frame) - tags_not_in_question.count(0), ' out of ', len(data_frame)

    plt.hist(tags_not_in_question)
    plt.title('tags missing from question text in {} corpus'.format(type_name if type_name else 'whole'))
    plt.xlabel('amount of tags')
    plt.show()

    print len(tags_not_in), ' of such tags: ', tags_not_in
    print 'tags containing "-" out of those: ', len([t for t in tags_not_in if '-' in t])


def write_question_num_per_tag(data, outfile=None, num=100, type_name=None):
    questions_per_tag = {}
    for col in list(tags_set):
        if sum(data[col].values) > 0:
            questions_per_tag[col] = sum(data[col].values)
    questions_per_tag_sorted = sorted(questions_per_tag.items(), key=operator.itemgetter(1), reverse=True)
    if outfile:
        with open(outfile, 'w') as f:
            for item in questions_per_tag_sorted:
                f.write('{} : {}\n'.format(item[0], item[1]))
    to_check = [item[0] for item in questions_per_tag_sorted[0:num]]

    t = [x[0] for x in questions_per_tag_sorted[0:50]]
    f = [x[1] for x in questions_per_tag_sorted[0:50]]
    frqplot(t, f, 'tag freq. in '.format(type_name))
    return to_check


if __name__ == '__main__':

    # VARIABLES
    DIR = '/home/christina/Documents/Thesis/data/askubuntu/additional/'
    NUM = 50
    # -----------------------------------------------------------------

    df = read_df(os.path.join(DIR, 'data_frame_corpus_str.csv'))
    df = df.fillna(u'')
    print 'num of questions ', df.shape[0]

    tags_set = set(list(df)) - {'id', 'title', 'body', 'tags', 'type', 'body_truncated'}
    print len(tags_set), ' tags: ', tags_set

    """--------------------------------- EVAL VALID ----------------------------------------------------------------"""
    questions_per_tag = {}
    data_train = df[df['type'] == 'train']
    data_eval = df[df['type'] != 'train']
    for col in list(tags_set):
        if 50 <= sum(data_train[col].values) <= 10000 and sum(data_eval[col].values) >= 5:
            questions_per_tag[col] = (sum(data_train[col].values), sum(data_eval[col].values))
    print questions_per_tag
    with open(os.path.join(DIR, 'tags_stats', 'valid_eval_tags.txt'), 'w') as f:
        for item in questions_per_tag.iteritems():
            f.write('{} : {}\n'.format(item[0], item[1]))
    valid_eval_tags = [k for k, v in questions_per_tag.iteritems()]
    pickle.dump(valid_eval_tags, open(os.path.join(DIR, 'tags_files', 'valid_eval_tags.p'), 'wb'))

    for typename, group_df in df.groupby('type'):
        no_labels_cases_if_valid_eval_tags_selected = np.array(group_df[valid_eval_tags].values)
        no_labels_cases_if_valid_eval_tags_selected = np.sum(no_labels_cases_if_valid_eval_tags_selected, 1)
        no_labels_cases_if_valid_eval_tags_selected = np.sum((no_labels_cases_if_valid_eval_tags_selected == 0).astype(np.int32))
        print no_labels_cases_if_valid_eval_tags_selected, ' with no labels in ', typename

    exit()
    """--------------------------------- EVAL VALID ----------------------------------------------------------------"""

    # """--------------------------------- TRAIN VALID ----------------------------------------------------------------"""
    # questions_per_tag = {}
    # data_train = df[df['type'] == 'train']
    # data_eval = df[df['type'] != 'train']
    # for col in list(tags_set):
    #     if 50 <= sum(data_train[col].values) <= 10000:
    #         questions_per_tag[col] = (sum(data_train[col].values), sum(data_eval[col].values))
    # print questions_per_tag
    # with open(os.path.join(DIR, 'tags_stats', 'valid_train_tags.txt'), 'w') as f:
    #     for item in questions_per_tag.iteritems():
    #         f.write('{} : {}\n'.format(item[0], item[1]))
    # valid_train_tags = [k for k, v in questions_per_tag.iteritems()]
    # pickle.dump(valid_train_tags, open(os.path.join(DIR, 'tags_files', 'valid_train_tags.p'), 'wb'))
    #
    # for typename, group_df in df.groupby('type'):
    #     no_labels_cases_if_valid_train_tags_selected = np.array(group_df[valid_train_tags].values)
    #     no_labels_cases_if_valid_train_tags_selected = np.sum(no_labels_cases_if_valid_train_tags_selected, 1)
    #     no_labels_cases_if_valid_train_tags_selected = np.sum((no_labels_cases_if_valid_train_tags_selected == 0).astype(np.int32))
    #     print no_labels_cases_if_valid_train_tags_selected, ' with no labels in ', typename
    #
    # exit()
    # """--------------------------------- TRAIN VALID ----------------------------------------------------------------"""

    # """--------------------------------- VALID    -------------------------------------------------------------------"""
    # questions_per_tag = {}
    # data_train = df[df['type'] == 'train']
    # data_eval = df[df['type'] != 'train']
    # for col in list(tags_set):
    #     if 50 <= sum(data_train[col].values) + sum(data_eval[col].values) <= 10000:
    #         questions_per_tag[col] = sum(data_train[col].values) + sum(data_eval[col].values)
    # print questions_per_tag
    # with open(os.path.join(DIR, 'tags_stats', 'valid_tags.txt'), 'w') as f:
    #     for item in questions_per_tag.iteritems():
    #         f.write('{} : {}\n'.format(item[0], item[1]))
    # valid_tags = [k for k, v in questions_per_tag.iteritems()]
    # pickle.dump(valid_tags, open(os.path.join(DIR, 'tags_files', 'valid_tags.p'), 'wb'))
    #
    # for typename, group_df in df.groupby('type'):
    #     no_labels_cases_if_valid_tags_selected = np.array(group_df[valid_tags].values)
    #     no_labels_cases_if_valid_tags_selected = np.sum(no_labels_cases_if_valid_tags_selected, 1)
    #     no_labels_cases_if_valid_tags_selected = np.sum((no_labels_cases_if_valid_tags_selected == 0).astype(np.int32))
    #     print no_labels_cases_if_valid_tags_selected, ' with no labels in ', typename
    #
    # exit()
    # """--------------------------------- VALID    -------------------------------------------------------------------"""

    # """--------------------------------- ABOVE 20 -------------------------------------------------------------------"""
    # questions_per_tag = {}
    # data_train = df[df['type'] == 'train']
    # data_eval = df[df['type'] != 'train']
    # for col in list(tags_set):
    #     if sum(data_train[col].values) >= 20 and sum(data_eval[col].values) >= 10:
    #         questions_per_tag[col] = (sum(data_train[col].values), sum(data_eval[col].values))
    # print questions_per_tag
    # with open(os.path.join(DIR, 'tags_stats', 'above20tags.txt'), 'w') as f:
    #     for item in questions_per_tag.iteritems():
    #         f.write('{} : {}\n'.format(item[0], item[1]))
    # above20tags = [k for k, v in questions_per_tag.iteritems()]
    # pickle.dump(above20tags, open(os.path.join(DIR, 'tags_files', 'above20tags.p'), 'wb'))
    #
    # for typename, group_df in df.groupby('type'):
    #     no_labels_cases_if_above20_selected = np.array(group_df[above20tags].values)
    #     no_labels_cases_if_above20_selected = np.sum(no_labels_cases_if_above20_selected, 1)
    #     no_labels_cases_if_above20_selected = np.sum((no_labels_cases_if_above20_selected == 0).astype(np.int32))
    #     print no_labels_cases_if_above20_selected, ' with no labels in ', typename
    #
    # exit()
    # """--------------------------------- ABOVE 20 -------------------------------------------------------------------"""

    hist_num_of_tags(df,)
    exit()
    tags_not_in_questions(df,)

    commonsname = os.path.join(DIR, 'top{}_common_tags.p'.format(NUM))
    corpussname = os.path.join(DIR, 'top{}_corpus_tags.p'.format(NUM))
    trainsname = os.path.join(DIR, 'top{}_train_tags.p'.format(NUM))

    try:
        common_selected = pickle.load(open(commonsname, 'rb'))
        corpus_selected = pickle.load(open(corpussname, 'rb'))
        train_selected = pickle.load(open(trainsname, 'rb'))
    except:

        top_tags = dict()
        top_tags['corpus'] = write_question_num_per_tag(
            df, outfile=os.path.join(DIR, 'corpus_{}tags_stats.txt'.format(NUM)), num=NUM, type_name='corpus')

        for typename, group_df in df.groupby('type'):
            top_tags[typename] = write_question_num_per_tag(
                group_df, os.path.join(DIR, '{}_{}tags_stats.txt'.format(typename, NUM)), NUM, typename)

        # After taking top from eval, tags not taken from train/corpus
        print '{} {}'.format(
            len(set(top_tags['train']) - set(top_tags['dev']) - set(top_tags['test'])),
            len(set(top_tags['corpus']) - set(top_tags['dev']) - set(top_tags['test']))
        )
        # After taking top from train/corpus, tags not taken from eval
        print '{} {}'.format(
            len((set(top_tags['test']) | set(top_tags['dev'])) - set(top_tags['train'])),
            len((set(top_tags['test']) | set(top_tags['dev'])) - set(top_tags['corpus']))
        )
        # common tags between top in tags dev and train/corpus
        print '{} {}'.format(
            len(set(top_tags['test']) & set(top_tags['dev']) & set(top_tags['train'])),
            len(set(top_tags['test']) & set(top_tags['dev']) & set(top_tags['corpus']))
        )
        print '\n\n'

        common_selected = list(set(top_tags['test']) & set(top_tags['dev']) & set(top_tags['train']))
        corpus_selected = top_tags['corpus']
        train_selected = top_tags['train']
        del top_tags
        pickle.dump(common_selected, open(commonsname, 'wb'))
        pickle.dump(corpus_selected, open(corpussname, 'wb'))
        pickle.dump(train_selected, open(trainsname, 'wb'))

    print 'common selected ', common_selected
    print 'corpus selected ', corpus_selected
    print 'train selected ', train_selected

    for typename, group_df in df.groupby('type'):
        no_labels_cases_if_common_selected = np.array(group_df[common_selected].values)
        no_labels_cases_if_common_selected = np.sum(no_labels_cases_if_common_selected, 1)
        no_labels_cases_if_common_selected = np.sum((no_labels_cases_if_common_selected == 0).astype(np.int32))
        print no_labels_cases_if_common_selected, ' with no labels in ', typename
        no_labels_cases_if_corpus_selected = np.array(group_df[corpus_selected].values)
        no_labels_cases_if_corpus_selected = np.sum(no_labels_cases_if_corpus_selected, 1)
        no_labels_cases_if_corpus_selected = np.sum((no_labels_cases_if_corpus_selected == 0).astype(np.int32))
        print no_labels_cases_if_corpus_selected, ' with no labels in ', typename
        no_labels_cases_if_train_selected = np.array(group_df[corpus_selected].values)
        no_labels_cases_if_train_selected = np.sum(no_labels_cases_if_train_selected, 1)
        no_labels_cases_if_train_selected = np.sum((no_labels_cases_if_train_selected == 0).astype(np.int32))
        print no_labels_cases_if_train_selected, ' with no labels in ', typename
