import matplotlib.pyplot as plt
import operator
from utils import read_questions_with_tags, read_df, store_df
import pickle
import os
import numpy as np


def hist_num_of_tags():
    num_of_tags = [len(q[3]) for q in Q]

    plt.hist(num_of_tags)
    plt.title('tags per question')
    plt.xlabel('amount of tags')
    plt.show()

    print 'tags: ', tags_set
    print 'num of tags: ', len(tags_set)

    print 'num of questions with 0 tags ', num_of_tags.count(0)
    print 'avg, min, max number of tags per question ', sum(num_of_tags)/(len(Q)+0.), min(num_of_tags), max(num_of_tags)


def tags_not_in_questions():
    tags_not_in_question = []
    tags_not_in = set()
    for q in Q:
        ans = 0
        for t in q[3]:
            if t not in q[1] and t not in q[2]:
                ans += 1
                tags_not_in.update({t})
        tags_not_in_question.append(ans)

    print 'some tags not in questions happens in ', len(Q) - tags_not_in_question.count(0), ' out of ', len(Q)

    plt.hist(tags_not_in_question)
    plt.title('tags missing from question text')
    plt.xlabel('amount of tags')
    plt.show()

    print 'set of such tags: ', tags_not_in
    print 'num of tags not in question: ', len(tags_not_in)
    print 'tags containing "-" out of those: ', len([t for t in tags_not_in if '-' in t])


# todo: SKIP TAGS WITH FREQUENCY < 5
def write_question_num_per_tag(data, outfile=None, num=100, freq=False):
    questions_per_tag = {}
    for col in columns[4:]:
        if sum(data[col].values) > 0:
            questions_per_tag[col] = sum(data[col].values)
    questions_per_tag_sorted = sorted(questions_per_tag.items(), key=operator.itemgetter(1), reverse=True)
    if outfile:
        with open(outfile, 'w') as f:
            for item in questions_per_tag_sorted:
                f.write('{} : {}\n'.format(item[0], item[1]))
    to_check = [item[0] for item in questions_per_tag_sorted[0:num]]
    if freq:
        return questions_per_tag_sorted[0:num]
    return to_check


if __name__ == '__main__':

    Q = list(read_questions_with_tags('/home/christina/Documents/Thesis/data/askubuntu/texts_raw_with_tags.txt'))
    print 'num of questions ', len(Q)

    tags_set = set()
    for q in Q:
        tags_set.update(set(q[3]))

    # hist_num_of_tags()
    # tags_not_in_questions()

    columns = ['id', 'title', 'body', 'tags'] + [tag for tag in tags_set]

    df = read_df('data.csv')

    NUM = 300
    NAME = 'common_selected_{}.p'.format(NUM)
    NAME2 = 'all_selected_{}.p'.format(NUM)

    if os.path.isfile(NAME) and os.path.isfile(NAME2):
        common_selected = pickle.load(open(NAME, 'rb'))
        all_selected = pickle.load(open(NAME2, 'rb'))
    else:

        lists = dict()
        # lists['all'] = write_question_num_per_tag(df, num=NUM)
        for type_name, group_df in df.groupby('type'):
            # write_question_num_per_tag(group_df, 'data_stats_{}.txt'.format(typename))
            lists[type_name] = write_question_num_per_tag(group_df, num=NUM)

        print len(set(lists['train']) - set(lists['dev']) - set(lists['test']))
        print len((set(lists['test']) | set(lists['dev'])) - set(lists['train']))
        print len(set(lists['test']) & set(lists['dev']) & set(lists['train']))
        # print len(set(lists['test']) & set(lists['dev']))
        print '\n\n'
        # print len((set(lists['test']) | set(lists['dev'])) - set(lists['all']))
        # print len(set(lists['test']) & set(lists['dev']) & set(lists['all']))

        common_selected = list(set(lists['test']) & set(lists['dev']) & set(lists['train']))
        all_selected = lists['train']
        del lists
        pickle.dump(common_selected, open(NAME, 'wb'))
        pickle.dump(all_selected, open(NAME2, 'wb'))

        for type_name, group_df in df.groupby('type'):
            values1 = np.array(group_df[common_selected].values)
            values1 = np.sum(values1, 1)
            values1 = np.sum((values1 == 0).astype(np.int32))
            print values1, ' with no labels in ', type_name
            values2 = np.array(group_df[all_selected].values)
            values2 = np.sum(values2, 1)
            values2 = np.sum((values2 == 0).astype(np.int32))
            print values2, ' with no labels in ', type_name

    print 'common selected ', common_selected
    print 'all selected ', all_selected

    from freq_plot import frqplot
    r = write_question_num_per_tag(df, num=50, freq=True)
    t = [x[0] for x in r]
    f = [x[1] for x in r]
    frqplot(t, f)
