from utils import read_df, read_eval_rows, store_df
import matplotlib.pyplot as plt


def get_eval_ids(eval_rows):
    q_ids_in_eval = set()
    for x in eval_rows:
        q_ids_in_eval.update({x[0]} | set(x[2]))
    print len(q_ids_in_eval), ' ids appearing in evaluation file: ', q_ids_in_eval
    return q_ids_in_eval


def get_eval_ids_per_train_rows(train_rows, q_ids_in_eval):
    eval_ids_per_train_row = []
    for x in train_rows:
        t_ids = {x[0]} | set(x[2])
        eval_ids_per_train_row.append(len(q_ids_in_eval & t_ids))
    return eval_ids_per_train_row


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


def remake_files_for_qa():
    pass


if __name__ == '__main__':
    df = read_df('data.csv')
    print 'total ids: ', df.shape[0]

    E = read_eval_rows('/home/christina/Documents/Thesis/data/askubuntu/test.txt')
    test_ids = get_eval_ids(E)
    E = read_eval_rows('/home/christina/Documents/Thesis/data/askubuntu/dev.txt')
    dev_ids = get_eval_ids(E)

    eval_ids = test_ids | dev_ids
    print 'total eval ids: ', len(eval_ids)

    T = read_eval_rows('/home/christina/Documents/Thesis/data/askubuntu/train_random.txt')
    eval_ids_per_train_rows = get_eval_ids_per_train_rows(T, eval_ids)
    plt.hist(eval_ids_per_train_rows)
    plt.title('eval ids per train row')
    plt.show()

    # ASSIGN CORRECT 'train' 'test' 'dev'
    df = make_data_frame_for_tag_training(df, list(test_ids), list(dev_ids))
    store_df(df, 'data.csv')
