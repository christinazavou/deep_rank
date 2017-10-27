import os

os.system("python qr/main.py --corpus /home/christinagzavou/Thesis/data/texts_raw_with_tags_str.txt "
          "--train /home/christinagzavou/Thesis/data/train_random.txt --test /home/christinagzavou/Thesis/data/test/txt"
          "--dev /home/christinagzavou/Thesis/data/dev.txt"
          "--embeddings /home/christinagzavou/Thesis/data/vectors_pruned.200.txt.gz"
          "--hidden_dim 400 --dropout 0.2 --reweight 0 --average 2 --layer bigru --batch_size 40 "
          "--save_dir /home/christinagzavou/Thesis/models/mine_qr/all_train/BIGRU400maxpoolrew0drop20concat1static/run2"
          " > /home/christinagzavou/Thesis/models/mine_qr/all_train/BIGRU400maxpoolrew0drop20concat1static/run2/train.txt")


os.system("python qr/main.py --corpus /home/christinagzavou/Thesis/data/texts_raw_with_tags_str.txt "
          "--train /home/christinagzavou/Thesis/data/train_random.txt --test /home/christinagzavou/Thesis/data/test/txt"
          "--dev /home/christinagzavou/Thesis/data/dev.txt"
          "--embeddings /home/christinagzavou/Thesis/data/vectors_pruned.200.txt.gz"
          "--hidden_dim 400 --dropout 0.2 --reweight 0 --average 2 --layer bigru --batch_size 40 "
          "--save_dir /home/christinagzavou/Thesis/models/mine_qr/all_train/BIGRU400maxpoolrew0drop20concat1useemb0/run1"
          " > /home/christinagzavou/Thesis/models/mine_qr/all_train/BIGRU400maxpoolrew0drop20concat1useemb0/run1/train.txt")


os.system("python qr/main.py --corpus /home/christinagzavou/Thesis/data/texts_raw_with_tags_str.txt "
          "--train /home/christinagzavou/Thesis/data/train_random.txt --test /home/christinagzavou/Thesis/data/test/txt"
          "--dev /home/christinagzavou/Thesis/data/dev.txt"
          "--embeddings /home/christinagzavou/Thesis/data/vectors_pruned.200.txt.gz"
          "--hidden_dim 400 --dropout 0.2 --reweight 0 --average 2 --layer bigru --batch_size 40 "
          "--save_dir /home/christinagzavou/Thesis/models/mine_qr/all_train/BIGRU400maxpoolrew0drop20concat1useemb0/run2"
          " > /home/christinagzavou/Thesis/models/mine_qr/all_train/BIGRU400maxpoolrew0drop20concat1useemb0/run2/train.txt")


