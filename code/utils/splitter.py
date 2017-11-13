import matplotlib.pyplot as plt
import pandas as pd


def str2int_list(str_list):
    result = []
    try:
        result = [int(item) for item in str_list]
    except:
        # print('except: {}\n'.format(str_list))
        pass
    return result


def read_eval_rows(eval_file):  # test or dev
    with open(eval_file) as f:
        questions = f.readlines()

        for line in questions:
            if len(line.split('\t')) == 4:
                q_id, q_ids_similar, q_ids_candidates, q_bm25_candidates = \
                    line.decode('utf-8').split(u'\t')
                q_ids_similar = q_ids_similar.split(' ')
                q_ids_candidates = q_ids_candidates.split(' ')
                q_ids_similar = str2int_list(q_ids_similar)
                q_ids_candidates = str2int_list(q_ids_candidates)
            else:
                q_id, q_ids_similar, q_ids_candidates = \
                    line.decode('utf-8').split(u'\t')
                q_ids_similar = q_ids_similar.split(' ')
                q_ids_candidates = q_ids_candidates.split(' ')
                q_ids_similar = str2int_list(q_ids_similar)
                q_ids_candidates = str2int_list(q_ids_candidates)
            yield int(q_id), q_ids_similar, q_ids_candidates


def read_df(df_file, chunk_size=None, read_columns=None):
    if '.csv' in df_file:
        if chunk_size:
            return pd.read_csv(df_file, encoding='utf8', index_col=0, chunksize=chunk_size)
        else:
            if read_columns:
                return pd.read_csv(df_file, encoding='utf8', index_col=0, usecols=read_columns)
            else:
                return pd.read_csv(df_file, encoding='utf8', index_col=0)
    elif '.p' in df_file:
        if read_columns:
            return pd.read_pickle(df_file)[read_columns]
        else:
            return pd.read_pickle(df_file)
    else:
        raise Exception(' unknown pandas file {}'.format(df_file))


def get_eval_ids(eval_rows):
    q_ids_in_eval = set()
    for x in eval_rows:
        q_ids_in_eval.update({x[0]} | set(x[2]))
    print len(q_ids_in_eval), ' ids appearing in evaluation file: ', q_ids_in_eval
    return q_ids_in_eval


def get_eval_ids_per_train_rows(train_rows, q_ids_in_eval):
    new_train_rows = []
    eval_ids_as_train_candidates = 0
    eval_ids_as_train_similar = 0
    eval_ids_as_train_queries = 0
    for x in train_rows:
        if x[0] in q_ids_in_eval:
            eval_ids_as_train_queries += 1
            print 'x[0]: ', x
        if len(q_ids_in_eval & set(x[1])) > 0:
            eval_ids_as_train_similar += len(q_ids_in_eval & set(x[1]))
        if len(q_ids_in_eval & set(x[2])) > 0:
            eval_ids_as_train_candidates += len(q_ids_in_eval & set(x[2]))
        if (x[0] in q_ids_in_eval) or (len(x[1]) - len(q_ids_in_eval & set(x[1])) == 0) or (len(x[2]) - len(q_ids_in_eval & set(x[2])) < 20):
                pass
        else:
            new_similar = list(set(x[1]) - q_ids_in_eval)
            new_candidate = list(set(x[2]) - q_ids_in_eval)
            new_x = (x[0], new_similar, new_candidate)
            if len(new_candidate) < len(x[2]):
                print new_x
            if len(new_similar) < len(x[1]):
                print new_x
            new_train_rows.append(new_x)
    print 'found ', eval_ids_as_train_queries, ' eval ids as train queries '
    print 'found ', eval_ids_as_train_candidates, ' eval ids as train candidate '
    print 'found ', eval_ids_as_train_similar, ' eval ids as train similar '
    print 'init train queries: ', len(train_rows)
    print 'new train rows ', len(new_train_rows)
    return new_train_rows


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


# def remake_training_file_for_question_ranking(train_rows, q_ids_in_eval, out_file):
#     with open(out_file, 'w') as f:
#         for x in train_rows:
#             if x[0] in q_ids_in_eval:
#                 continue
#             q_ids_similar = [q_id_sim for q_id_sim in x[1] if q_id_sim not in q_ids_in_eval]
#             q_ids_candidates = [q_id_cand for q_id_cand in x[2] if q_id_cand not in q_ids_in_eval]
#             q_ids_similar = " ".join([str(q) for q in q_ids_similar])
#             q_ids_candidates = " ".join([str(q) for q in q_ids_candidates])
#             f.write('{}\t{}\t{}\n'.format(str(x[0]), q_ids_similar, q_ids_candidates))


def remake_training_file_for_question_ranking(new_train_rows, out_file):
    with open(out_file, 'w') as f:
        for x in new_train_rows:
            q_ids_similar = " ".join([str(q) for q in x[1]])
            q_ids_candidates = " ".join([str(q) for q in x[2]])
            f.write('{}\t{}\t{}\n'.format(str(x[0]), q_ids_similar, q_ids_candidates))


if __name__ == '__main__':
    df = read_df('/home/christina/Documents/Thesis/data/askubuntu/additional/data_frame_corpus_str.csv')
    print 'total ids: ', df.shape[0]

    E = read_eval_rows('/home/christina/Documents/Thesis/data/askubuntu/test.txt')
    test_ids = get_eval_ids(E)
    E = read_eval_rows('/home/christina/Documents/Thesis/data/askubuntu/dev.txt')
    dev_ids = get_eval_ids(E)

    eval_ids = test_ids | dev_ids
    print 'total eval ids: ', len(eval_ids)

    T = list(read_eval_rows('/home/christina/Documents/Thesis/data/askubuntu/train_random.txt'))  # TO BE REUSED
    newT = get_eval_ids_per_train_rows(T, eval_ids)
    exit()
    # get_eval_ids_per_train_rows(newT, eval_ids)

    # # ASSIGN CORRECT 'train' 'test' 'dev'
    # df = make_data_frame_for_tag_training(df, list(test_ids), list(dev_ids))
    # store_df(df, '/home/christina/Documents/Thesis/data/askubuntu/data_frame_corpus.csv')

    # remake_training_file_for_question_ranking(
    #     T, eval_ids, '/home/christina/Documents/Thesis/data/askubuntu/additional/train_random_removed_eval.txt'
    # )
    remake_training_file_for_question_ranking(
        newT, '/home/christina/Documents/Thesis/data/askubuntu/additional/train_random_correct.txt'
    )
