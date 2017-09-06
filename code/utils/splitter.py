from utils import read_df, read_eval_rows, store_df
import matplotlib.pyplot as plt


def get_eval_ids(eval_rows):
    q_ids_in_eval = set()
    for x in eval_rows:
        q_ids_in_eval.update({x[0]} | set(x[2]))
    print len(q_ids_in_eval), ' ids appearing in evaluation file: ', q_ids_in_eval
    return q_ids_in_eval


def get_eval_ids_per_train_rows(train_rows, q_ids_in_eval):
    eval_ids_per_train_row_candidates = []
    eval_ids_per_train_row_similar = []
    eval_ids_as_train_queries = 0
    for x in train_rows:
        if x[0] in q_ids_in_eval:
            eval_ids_as_train_queries += 1
        t_ids = set(x[2])
        eval_ids_per_train_row_candidates.append(len(q_ids_in_eval & t_ids))
        t_ids = set(x[1])
        eval_ids_per_train_row_similar.append(len(q_ids_in_eval & t_ids))
    print 'found ', eval_ids_as_train_queries, ' eval ids as train queries '
    print 'init train queries: ', len(train_rows)
    return eval_ids_per_train_row_candidates, eval_ids_per_train_row_similar


def make_data_frame_for_tag_training(data_frame, test, dev):
    df['type'] = 'train'
    for q_id in test:
        idx = data_frame[data_frame['id'] == q_id].index[0]
        data_frame.set_value(idx, 'type', 'test')  # few ids common with dev will be assigned to dev
    for q_id in dev:
        idx = data_frame[data_frame['id'] == q_id].index[0]
        data_frame.set_value(idx, 'type', 'dev')

    print df[df['type'] == 'test'].shape
    print df[df['type'] == 'dev'].shape
    print df[df['type'] == 'train'].shape

    return data_frame


def remake_training_file_for_question_ranking(train_rows, q_ids_in_eval, out_file):
    with open(out_file, 'w') as f:
        for x in train_rows:
            if x[0] in q_ids_in_eval:
                continue
            q_ids_similar = [q_id_sim for q_id_sim in x[1] if q_id_sim not in q_ids_in_eval]
            q_ids_candidates = [q_id_cand for q_id_cand in x[2] if q_id_cand not in q_ids_in_eval]
            q_ids_similar = " ".join([str(q) for q in q_ids_similar])
            q_ids_candidates = " ".join([str(q) for q in q_ids_candidates])
            f.write('{}\t{}\t{}\n'.format(str(x[0]), q_ids_similar, q_ids_candidates))


if __name__ == '__main__':
    df = read_df('/home/christina/Documents/Thesis/data/askubuntu/additional/data_frame_corpus.csv')
    print 'total ids: ', df.shape[0]

    E = read_eval_rows('/home/christina/Documents/Thesis/data/askubuntu/test.txt')
    test_ids = get_eval_ids(E)
    E = read_eval_rows('/home/christina/Documents/Thesis/data/askubuntu/dev.txt')
    dev_ids = get_eval_ids(E)

    eval_ids = test_ids | dev_ids
    print 'total eval ids: ', len(eval_ids)

    T = list(read_eval_rows('/home/christina/Documents/Thesis/data/askubuntu/train_random.txt'))  # TO BE REUSED
    eval_ids_per_train_rows_candidates, eval_ids_per_train_rows_similar = get_eval_ids_per_train_rows(T, eval_ids)
    plt.hist(eval_ids_per_train_rows_candidates)
    plt.title('eval ids per train row candidate')
    plt.show()
    plt.hist(eval_ids_per_train_rows_similar)
    plt.title('eval ids per train row similar')
    plt.show()

    # # ASSIGN CORRECT 'train' 'test' 'dev'
    # df = make_data_frame_for_tag_training(df, list(test_ids), list(dev_ids))
    # store_df(df, '/home/christina/Documents/Thesis/data/askubuntu/data_frame_corpus.csv')

    remake_training_file_for_question_ranking(
        T, eval_ids, '/home/christina/Documents/Thesis/data/askubuntu/additional/train_random_removed_eval.txt'
    )
