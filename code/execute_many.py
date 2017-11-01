import os


os.system("python qr/main.py --corpus /home/christinagzavou/Thesis/data/texts_raw_fixed.txt "
          "--train /home/christinagzavou/Thesis/data/train_random_removed_eval.txt "
          "--test /home/christinagzavou/Thesis/data/test.txt "
          "--dev /home/christinagzavou/Thesis/data/dev.txt "
          "--embeddings /home/christinagzavou/Thesis/data/vectors_pruned.200.txt.gz "
          "--hidden_dim 667 --dropout 0.2 --reweight 0 --average 0 --layer cnn --batch_size 40 --use_embeddings 0 "
          "--save_dir /home/christinagzavou/Thesis/models/mine_qr/reduce_train/CNN667maxpooldrop20rew0/run1 "
          "> /home/christinagzavou/Thesis/models/mine_qr/reduce_train/CNN667maxpooldrop20rew0/run1/train.txt")

os.system("python tags_prediction/main.py --corpus_w_tags /home/christinagzavou/Thesis/data/texts_raw_with_tags_str.txt "
          "--df_path /home/christinagzavou/Thesis/data/data_frame_corpus_str.csv --tags_file /home/christinagzavou/Thesis/data/valid_train_tags.p "
          "--embeddings /home/christinagzavou/Thesis/data/vectors_pruned.200.txt.gz "
          "--hidden_dim 667 --dropout 0.2 --reweight 0 --average 0 --layer cnn --batch_size 128 --trainable 0 "
          "--load_pre_trained_part /home/christinagzavou/Thesis/models/mine_qr/reduce_train/CNN667maxpooldrop20rew0static/best_model.pkl.gz "
          "--trainable_encoder 1 "
          "--save_dir /home/christinagzavou/Thesis/models/tp_pretrained_on_qr/reduce_train/CNN667maxpoolrew0drop20staticemb/run1 "
          "> /home/christinagzavou/Thesis/models/tp_pretrained_on_qr/reduce_train/CNN667maxpoolrew0drop20staticemb/run1/train.txt")
