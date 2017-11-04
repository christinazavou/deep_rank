import os

os.system("python tags_prediction/main.py "
          "--corpus_w_tags /home/christina/Documents/Thesis/data/askubuntu/additional/texts_raw_with_tags_str.txt "
          "--df_path /home/christina/Documents/Thesis/data/askubuntu/additional/data_frame_corpus_str.csv "
          "--tags_file /home/christina/Documents/Thesis/data/askubuntu/additional/tags_files/valid_train_tags.p "
          "--embeddings /home/christina/Documents/Thesis/data/askubuntu/vector/vectors_pruned.200.txt.gz "
          "--hidden_dim 280 --dropout 0.3 --reweight 0 --average 2 --layer gru --batch_size 128 --trainable 0 "
          "--cross_val 1 "
          "--save_dir /media/christina/Data/Thesis/models/askubuntu/tags_prediction/R@10/corpus908/cross_val/GRU/run1 "
          "> /media/christina/Data/Thesis/models/askubuntu/tags_prediction/R@10/corpus908/cross_val/GRU/run1/train.txt")

os.system("python tags_prediction/main.py "
          "--corpus_w_tags /home/christina/Documents/Thesis/data/askubuntu/additional/texts_raw_with_tags_str.txt "
          "--df_path /home/christina/Documents/Thesis/data/askubuntu/additional/data_frame_corpus_str.csv "
          "--tags_file /home/christina/Documents/Thesis/data/askubuntu/additional/tags_files/valid_train_tags.p "
          "--embeddings /home/christina/Documents/Thesis/data/askubuntu/vector/vectors_pruned.200.txt.gz "
          "--hidden_dim 280 --dropout 0.3 --reweight 0 --average 2 --layer gru --batch_size 128 --trainable 0 "
          "--cross_val 1 "
          "--save_dir /media/christina/Data/Thesis/models/askubuntu/tags_prediction/R@10/corpus908/cross_val/GRU/run2 "
          "> /media/christina/Data/Thesis/models/askubuntu/tags_prediction/R@10/corpus908/cross_val/GRU/run2/train.txt")

os.system("python tags_prediction/main.py "
          "--corpus_w_tags /home/christina/Documents/Thesis/data/askubuntu/additional/texts_raw_with_tags_str.txt "
          "--df_path /home/christina/Documents/Thesis/data/askubuntu/additional/data_frame_corpus_str.csv "
          "--tags_file /home/christina/Documents/Thesis/data/askubuntu/additional/tags_files/valid_train_tags.p "
          "--embeddings /home/christina/Documents/Thesis/data/askubuntu/vector/vectors_pruned.200.txt.gz "
          "--hidden_dim 240 --dropout 0.3 --reweight 0 --average 2 --layer lstm --batch_size 128 --trainable 0 "
          "--cross_val 1 "
          "--save_dir /media/christina/Data/Thesis/models/askubuntu/tags_prediction/R@10/corpus908/cross_val/LSTM/run1 "
          "> /media/christina/Data/Thesis/models/askubuntu/tags_prediction/R@10/corpus908/cross_val/LSTM/run1/train.txt")

os.system("python tags_prediction/main.py "
          "--corpus_w_tags /home/christina/Documents/Thesis/data/askubuntu/additional/texts_raw_with_tags_str.txt "
          "--df_path /home/christina/Documents/Thesis/data/askubuntu/additional/data_frame_corpus_str.csv "
          "--tags_file /home/christina/Documents/Thesis/data/askubuntu/additional/tags_files/valid_train_tags.p "
          "--embeddings /home/christina/Documents/Thesis/data/askubuntu/vector/vectors_pruned.200.txt.gz "
          "--hidden_dim 240 --dropout 0.3 --reweight 0 --average 2 --layer lstm --batch_size 128 --trainable 0 "
          "--cross_val 1 "
          "--save_dir /media/christina/Data/Thesis/models/askubuntu/tags_prediction/R@10/corpus908/cross_val/LSTM/run2 "
          "> /media/christina/Data/Thesis/models/askubuntu/tags_prediction/R@10/corpus908/cross_val/LSTM/run2/train.txt")

# os.system("python qr/main.py --corpus /home/christinagzavou/Thesis/data/texts_raw_fixed.txt "
#           "--train /home/christinagzavou/Thesis/data/train_random_removed_eval.txt "
#           "--test /home/christinagzavou/Thesis/data/test.txt "
#           "--dev /home/christinagzavou/Thesis/data/dev.txt "
#           "--embeddings /home/christinagzavou/Thesis/data/vectors_pruned.200.txt.gz "
#           "--hidden_dim 667 --dropout 0.2 --reweight 0 --average 0 --layer cnn --batch_size 40 --use_embeddings 0 "
#           "--save_dir /home/christinagzavou/Thesis/models/mine_qr/reduce_train/CNN667maxpooldrop20rew0/run1 "
#           "> /home/christinagzavou/Thesis/models/mine_qr/reduce_train/CNN667maxpooldrop20rew0/run1/train.txt")
#
# os.system("python tags_prediction/main.py --corpus_w_tags /home/christinagzavou/Thesis/data/texts_raw_with_tags_str.txt "
#           "--df_path /home/christinagzavou/Thesis/data/data_frame_corpus_str.csv --tags_file /home/christinagzavou/Thesis/data/valid_train_tags.p "
#           "--embeddings /home/christinagzavou/Thesis/data/vectors_pruned.200.txt.gz "
#           "--hidden_dim 667 --dropout 0.2 --reweight 0 --average 0 --layer cnn --batch_size 128 --trainable 0 "
#           "--load_pre_trained_part /home/christinagzavou/Thesis/models/mine_qr/reduce_train/CNN667maxpooldrop20rew0static/best_model.pkl.gz "
#           "--trainable_encoder 1 "
#           "--save_dir /home/christinagzavou/Thesis/models/tp_pretrained_on_qr/reduce_train/CNN667maxpoolrew0drop20staticemb/run1 "
#           "> /home/christinagzavou/Thesis/models/tp_pretrained_on_qr/reduce_train/CNN667maxpoolrew0drop20staticemb/run1/train.txt")

# os.system("python multitask/main.py "
#           "--corpus_w_tags /home/christina/Documents/Thesis/data/askubuntu/additional/texts_raw_with_tags_str.txt "
#           "--tags_file /home/christina/Documents/Thesis/data/askubuntu/additional/tags_files/valid_train_tags.p "
#           "--embeddings /home/christina/Documents/Thesis/data/askubuntu/vector/vectors_pruned.200.txt.gz "
#           "--corpus /home/christina/Documents/Thesis/data/askubuntu/texts_raw_fixed.txt "
#           "--train /home/christina/Documents/Thesis/data/askubuntu/train_random.txt "
#           "--test /home/christina/Documents/Thesis/data/askubuntu/test.txt "
#           "--dev /home/christina/Documents/Thesis/data/askubuntu/dev.txt "
#           "--trainable 0 -d 240 --layer lstm --average 2 --reweight 0 --dropout 0.2 "
#           "--save_dir /media/christina/Data/Thesis/models/askubuntu/multitask/no_pretrain/corpus908/dev_MRR/LSTM240rew0maxpooldrop20static/run1 "
#           "> /media/christina/Data/Thesis/models/askubuntu/multitask/no_pretrain/corpus908/dev_MRR/LSTM240rew0maxpooldrop20static/run1/train.txt")
#
# os.system("python multitask/main.py "
#           "--corpus_w_tags /home/christina/Documents/Thesis/data/askubuntu/additional/texts_raw_with_tags_str.txt "
#           "--tags_file /home/christina/Documents/Thesis/data/askubuntu/additional/tags_files/valid_train_tags.p "
#           "--embeddings /home/christina/Documents/Thesis/data/askubuntu/vector/vectors_pruned.200.txt.gz "
#           "--corpus /home/christina/Documents/Thesis/data/askubuntu/texts_raw_fixed.txt "
#           "--train /home/christina/Documents/Thesis/data/askubuntu/train_random.txt "
#           "--test /home/christina/Documents/Thesis/data/askubuntu/test.txt "
#           "--dev /home/christina/Documents/Thesis/data/askubuntu/dev.txt "
#           "--trainable 0 -d 240 --layer lstm --average 2 --reweight 0 --dropout 0.2 --performance R@10 "
#           "--save_dir /media/christina/Data/Thesis/models/askubuntu/multitask/no_pretrain/corpus908/Rat10/LSTM240rew0maxpooldrop20static/run1 "
#           "> /media/christina/Data/Thesis/models/askubuntu/multitask/no_pretrain/corpus908/Rat10/LSTM240rew0maxpooldrop20static/run1/train.txt")
#
# os.system("python multitask/main.py "
#           "--corpus_w_tags /home/christina/Documents/Thesis/data/askubuntu/additional/texts_raw_with_tags_str.txt "
#           "--tags_file /home/christina/Documents/Thesis/data/askubuntu/additional/tags_files/valid_train_tags.p "
#           "--embeddings /home/christina/Documents/Thesis/data/askubuntu/vector/vectors_pruned.200.txt.gz "
#           "--corpus /home/christina/Documents/Thesis/data/askubuntu/texts_raw_fixed.txt "
#           "--train /home/christina/Documents/Thesis/data/askubuntu/train_random.txt "
#           "--test /home/christina/Documents/Thesis/data/askubuntu/test.txt "
#           "--dev /home/christina/Documents/Thesis/data/askubuntu/dev.txt "
#           "--trainable 0 -d 280 --layer gru --average 2 --reweight 0 --dropout 0.2 "
#           "--save_dir /media/christina/Data/Thesis/models/askubuntu/multitask/no_pretrain/corpus908/dev_MRR/GRU280rew0maxpooldrop20static/run1 "
#           "> /media/christina/Data/Thesis/models/askubuntu/multitask/no_pretrain/corpus908/dev_MRR/GRU280rew0maxpooldrop20static/run1/train.txt")
#
# os.system("python multitask/main.py "
#           "--corpus_w_tags /home/christina/Documents/Thesis/data/askubuntu/additional/texts_raw_with_tags_str.txt "
#           "--tags_file /home/christina/Documents/Thesis/data/askubuntu/additional/tags_files/valid_train_tags.p "
#           "--embeddings /home/christina/Documents/Thesis/data/askubuntu/vector/vectors_pruned.200.txt.gz "
#           "--corpus /home/christina/Documents/Thesis/data/askubuntu/texts_raw_fixed.txt "
#           "--train /home/christina/Documents/Thesis/data/askubuntu/train_random.txt "
#           "--test /home/christina/Documents/Thesis/data/askubuntu/test.txt "
#           "--dev /home/christina/Documents/Thesis/data/askubuntu/dev.txt "
#           "--trainable 0 -d 280 --layer gru --average 2 --reweight 0 --dropout 0.2 --performance R@10 "
#           "--save_dir /media/christina/Data/Thesis/models/askubuntu/multitask/no_pretrain/corpus908/Rat10/GRU280rew0maxpooldrop20static/run1 "
#           "> /media/christina/Data/Thesis/models/askubuntu/multitask/no_pretrain/corpus908/Rat10/GRU280rew0maxpooldrop20static/run1/train.txt")
#
