import os


# code_dir = os.path.dirname(os.path.realpath(__file__))
# os.system("python tags_prediction/main.py --corpus_w_tags /home/christina/Documents/Thesis/data/askubuntu/additional/"
#           "texts_raw_with_tags_str.txt --df_path /home/christina/Documents/Thesis/data/askubuntu/"
#           "additional/data_frame_corpus_str.csv --tags_file /home/christina/Documents/Thesis/data/askubuntu/"
#           "additional/tags_files/valid_train_tags.p --embeddings /home/christina/Documents/Thesis/data/askubuntu/"
#           "vector/vectors_pruned.200.txt.gz --save_dir {}/check_runs1 --max_epoch 1 > gianadoume1.txt".
#           format(code_dir))


os.system("python tags_prediction/main.py --corpus_w_tags /home/christina/Documents/Thesis/data/askubuntu/additional/"
          "texts_raw_with_tags_str.txt --df_path /home/christina/Documents/Thesis/data/askubuntu/"
          "additional/data_frame_corpus_str.csv --tags_file /home/christina/Documents/Thesis/data/askubuntu/"
          "additional/tags_files/valid_train_tags.p --embeddings /home/christina/Documents/Thesis/data/askubuntu/"
          "vector/vectors_pruned.200.txt.gz --load_pre_trained_part /media/christina/Data/Thesis/models/askubuntu/"
          "mine_qr/reduce_train/GRU280maxpooldrop20rew0/best_model.pkl.gz --hidden_dim 280 --dropout 0.2 --reweight 0 "
          "--average 2 --layer gru --batch_size 128 --save_dir /media/christina/Data/Thesis/models/askubuntu/"
          "tp_pretrained_on_qr/corpus908/GRU280maxpoolrew0drop20/run1 > /media/christina/Data/Thesis/models/askubuntu/"
          "tp_pretrained_on_qr/corpus908/GRU280maxpoolrew0drop20/run1/train.txt")

os.system("python tags_prediction/main.py --corpus_w_tags /home/christina/Documents/Thesis/data/askubuntu/additional/"
          "texts_raw_with_tags_str.txt --df_path /home/christina/Documents/Thesis/data/askubuntu/"
          "additional/data_frame_corpus_str.csv --tags_file /home/christina/Documents/Thesis/data/askubuntu/"
          "additional/tags_files/valid_train_tags.p --embeddings /home/christina/Documents/Thesis/data/askubuntu/"
          "vector/vectors_pruned.200.txt.gz --load_pre_trained_part /media/christina/Data/Thesis/models/askubuntu/"
          "mine_qr/reduce_train/GRU280maxpooldrop20rew0/best_model.pkl.gz --hidden_dim 280 --dropout 0.2 --reweight 0 "
          "--average 2 --layer gru --batch_size 128 --save_dir /media/christina/Data/Thesis/models/askubuntu/"
          "tp_pretrained_on_qr/corpus908/GRU280maxpoolrew0drop20/run2 > /media/christina/Data/Thesis/models/askubuntu/"
          "tp_pretrained_on_qr/corpus908/GRU280maxpoolrew0drop20/run2/train.txt")

os.system("python tags_prediction/main.py --corpus_w_tags /home/christina/Documents/Thesis/data/askubuntu/additional/"
          "texts_raw_with_tags_str.txt --df_path /home/christina/Documents/Thesis/data/askubuntu/"
          "additional/data_frame_corpus_str.csv --tags_file /home/christina/Documents/Thesis/data/askubuntu/"
          "additional/tags_files/valid_train_tags.p --embeddings /home/christina/Documents/Thesis/data/askubuntu/"
          "vector/vectors_pruned.200.txt.gz --load_pre_trained_part /media/christina/Data/Thesis/models/askubuntu/"
          "mine_qr/reduce_train/CNN667win3maxpooldrop20rew0/best_model.pkl.gz --hidden_dim 667 --dropout 0.2 "
          "--reweight 0 --average 0 --layer cnn --batch_size 128 --save_dir /media/christina/Data/Thesis/models/"
          "askubuntu/tp_pretrained_on_qr/corpus908/CNN_init_weights/run4 > /media/christina/Data/Thesis/models/"
          "askubuntu/tp_pretrained_on_qr/corpus908/CNN_init_weights/run4/train.txt")

os.system("python tags_prediction/main.py --corpus_w_tags /home/christina/Documents/Thesis/data/askubuntu/additional/"
          "texts_raw_with_tags_str.txt --df_path /home/christina/Documents/Thesis/data/askubuntu/"
          "additional/data_frame_corpus_str.csv --tags_file /home/christina/Documents/Thesis/data/askubuntu/"
          "additional/tags_files/valid_train_tags.p --embeddings /home/christina/Documents/Thesis/data/askubuntu/"
          "vector/vectors_pruned.200.txt.gz --load_pre_trained_part /media/christina/Data/Thesis/models/askubuntu/"
          "mine_qr/reduce_train/CNN667win3maxpooldrop20rew0/best_model.pkl.gz --hidden_dim 667 --dropout 0.2 "
          "--reweight 0 --average 0 --layer cnn --batch_size 128 --save_dir /media/christina/Data/Thesis/models/"
          "askubuntu/tp_pretrained_on_qr/corpus908/CNN_init_weights/run5 > /media/christina/Data/Thesis/models/"
          "askubuntu/tp_pretrained_on_qr/corpus908/CNN_init_weights/run5/train.txt")

os.system("python qr/main.py --corpus /home/christina/Documents/Thesis/data/askubuntu/texts_raw_fixed.txt --train "
          "/home/christina/Documents/Thesis/data/askubuntu/additional/train_random_removed_eval.txt --test /home/"
          "christina/Documents/Thesis/data/askubuntu/test.txt --dev /home/christina/Documents/Thesis/data/askubuntu/"
          "dev.txt --embeddings /home/christina/Documents/Thesis/data/askubuntu/vector/vectors_pruned.200.txt.gz "
          "--load_pre_trained_part /media/christina/Data/Thesis/models/askubuntu/tags_prediction/R@10/corpus908/"
          "CNN_maxpool/_with_init_weights/best_model.pkl.gz --hidden_dim 667 --dropout 0.2 --reweight 0 --average 0 "
          "--layer cnn --batch_size 40 --save_dir /media/christina/Data/Thesis/models/askubuntu/qr_pretrained_on_tp/"
          "reduce_train/corpus908/CNN_maxpool_win3_rew0drop20/run3 > /media/christina/Data/Thesis/models/askubuntu/"
          "qr_pretrained_on_tp/reduce_train/corpus908/CNN_maxpool_win3_rew0drop20/run3/train.txt")
