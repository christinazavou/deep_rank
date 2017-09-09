
import pickle
from utils import str2int_list, read_df
import numpy as np

tags_file = '/home/christina/Documents/Thesis/data/askubuntu/additional/tags_files/top100_corpus_tags.p'

ids_file = '/home/christina/Documents/Thesis/models/askubuntu/taolei_askubuntu14_norm_weight_average_20_240_lstm/dev_ids_tockeck.p'
eval_file = '/home/christina/Documents/Thesis/data/askubuntu/dev.txt'

ids_file = '/home/christina/Documents/Thesis/models/askubuntu/taolei_askubuntu14_norm_weight_average_20_240_lstm/test_ids_tocheck.p'
eval_file = '/home/christina/Documents/Thesis/data/askubuntu/test.txt'

ids_to_check = pickle.load(open(ids_file, 'rb'))
print len(ids_to_check)

with open(eval_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        qid, qidsim, qidscand, qidsscores = line.strip().split('\t')
        if int(qid) in ids_to_check:
            ids_to_check.extend(str2int_list(qidscand))

ids_to_check = list(set(ids_to_check))
print len(ids_to_check)

df = read_df('/home/christina/Documents/Thesis/data/askubuntu/additional/data_frame_corpus.csv')

label_tags = pickle.load(open(tags_file, 'rb'))

count = 0
for idx, row in df.iterrows():
    if int(row['id']) in ids_to_check:
        if (np.array(row[label_tags].values).astype(np.float32) == np.zeros(100)).all():
            count += 1
print count

