import os

os.system("python qr/main.py --corpus /home/christinagzavou/Thesis/data/texts_raw_with_tags_str.txt "
          "--train /home/christinagzavou/Thesis/data/train_random.txt --test /home/christinagzavou/Thesis/data/test/txt"
          "--dev /home/christinagzavou/Thesis/data/dev.txt"
          "--embeddings /home/christinagzavou/Thesis/data/vectors_pruned.200.txt.gz"
          "--hidden_dim 400 --dropout 0.2 --reweight 0 --average 2 --layer bigru --batch_size 40 "
          "--save_dir /home/christinagzavou/Thesis/models/mine_qr/all_train/BIGRU400maxpoolrew0drop20concat1static/run2"
          " > /home/christinagzavou/Thesis/models/mine_qr/all_train/BIGRU400maxpoolrew0drop20concat1static/run2/train.txt")

os.system("python tags_prediction/main.py --corpus_w_tags /home/christina/Documents/Thesis/data/askubuntu/additional/"
          "texts_raw_with_tags_str.txt "
          "--df_path /home/christina/Documents/Thesis/data/askubuntu/additional/data_frame_corpus_str.csv --tags_file "
          "/home/christina/Documents/Thesis/data/askubuntu/additional/tags_files/valid_train_tags.p"
          "--embeddings /home/christinagzavou/Thesis/data/vectors_pruned.200.txt.gz"
          "--hidden_dim 667 --dropout 0.2 --reweight 0 --average 0 --layer cnn --batch_size 128 --trainable 0 "
          "--save_dir /media/christina/Data/Thesis/models/askubuntu/tags_prediction/R@10/corpus908/"
          "CNN667win3maxpoolrew0drop20static/run1"
          " > /media/christina/Data/Thesis/models/askubuntu/tags_prediction/R@10/corpus908/"
          "CNN667win3maxpoolrew0drop20static/run1/train.txt")

