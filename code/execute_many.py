import os

os.system("python qr/main.py --corpus /home/christina/Documents/Thesis/data/askubuntu/texts_raw_fixed.txt "
          "--train /home/christina/Documents/Thesis/data/askubuntu/additional/train_random_removed_eval.txt "
          "--test /home/christina/Documents/Thesis/data/askubuntu/test.txt "
          "--dev /home/christina/Documents/Thesis/data/askubuntu/dev.txt "
          "--embeddings /home/christina/Documents/Thesis/data/askubuntu/vector/vectors_pruned.200.txt.gz "
          "--hidden_dim 667 --dropout 0.2 --reweight 0 --average 0 --layer cnn --batch_size 40 --trainable 0 "
          "--save_dir /media/christina/Data/Thesis/models/askubuntu/mine_qr/reduce_train/CNN667win3maxpoooldrop20rew0static/run3 "
          "> /media/christina/Data/Thesis/models/askubuntu/mine_qr/reduce_train/CNN667win3maxpoooldrop20rew0static/run3/train.txt")

os.system("python qr/main.py --corpus /home/christina/Documents/Thesis/data/askubuntu/texts_raw_fixed.txt "
          "--train /home/christina/Documents/Thesis/data/askubuntu/additional/train_random_removed_eval.txt "
          "--test /home/christina/Documents/Thesis/data/askubuntu/test.txt "
          "--dev /home/christina/Documents/Thesis/data/askubuntu/dev.txt "
          "--embeddings /home/christina/Documents/Thesis/data/askubuntu/vector/vectors_pruned.200.txt.gz "
          "--hidden_dim 667 --dropout 0.2 --reweight 0 --average 0 --layer cnn --batch_size 40 --trainable 0 "
          "--save_dir /media/christina/Data/Thesis/models/askubuntu/mine_qr/reduce_train/CNN667win3maxpoooldrop20rew0static/run4 "
          "> /media/christina/Data/Thesis/models/askubuntu/mine_qr/reduce_train/CNN667win3maxpoooldrop20rew0static/run4/train.txt")

os.system("python qr/main.py --corpus /home/christina/Documents/Thesis/data/askubuntu/texts_raw_fixed.txt "
          "--train /home/christina/Documents/Thesis/data/askubuntu/additional/train_random_removed_eval.txt "
          "--test /home/christina/Documents/Thesis/data/askubuntu/test.txt "
          "--dev /home/christina/Documents/Thesis/data/askubuntu/dev.txt "
          "--embeddings /home/christina/Documents/Thesis/data/askubuntu/vector/vectors_pruned.200.txt.gz "
          "--hidden_dim 667 --dropout 0.2 --reweight 0 --average 0 --layer cnn --batch_size 40 --trainable 0 "
          "--save_dir /media/christina/Data/Thesis/models/askubuntu/mine_qr/reduce_train/CNN667win3maxpoooldrop20rew0static/run5 "
          "> /media/christina/Data/Thesis/models/askubuntu/mine_qr/reduce_train/CNN667win3maxpoooldrop20rew0static/run5/train.txt")


# os.system("python tags_prediction/main.py --corpus_w_tags /home/christina/Documents/Thesis/data/askubuntu/additional/"
#           "texts_raw_with_tags_str.txt "
#           "--df_path /home/christina/Documents/Thesis/data/askubuntu/additional/data_frame_corpus_str.csv --tags_file "
#           "/home/christina/Documents/Thesis/data/askubuntu/additional/tags_files/valid_train_tags.p"
#           "--embeddings /home/christinagzavou/Thesis/data/vectors_pruned.200.txt.gz"
#           "--hidden_dim 667 --dropout 0.2 --reweight 0 --average 0 --layer cnn --batch_size 128 --trainable 0 "
#           "--save_dir /media/christina/Data/Thesis/models/askubuntu/tags_prediction/R@10/corpus908/"
#           "CNN667win3maxpoolrew0drop20static/run1"
#           " > /media/christina/Data/Thesis/models/askubuntu/tags_prediction/R@10/corpus908/"
#           "CNN667win3maxpoolrew0drop20static/run1/train.txt")
#
