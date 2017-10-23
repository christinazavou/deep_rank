import os
# import sys
# print os.environ['PYTHONPATH']
code_dir = os.path.dirname(os.path.realpath(__file__))
# os.system("cd tags_prediction")
# os.system("echo $(pwd)")

os.system("python tags_prediction/main.py --corpus_w_tags /home/christina/Documents/Thesis/data/askubuntu/additional/"
          "texts_raw_with_tags_str.txt --df_path /home/christina/Documents/Thesis/data/askubuntu/"
          "additional/data_frame_corpus_str.csv --tags_file /home/christina/Documents/Thesis/data/askubuntu/"
          "additional/tags_files/valid_train_tags.p --embeddings /home/christina/Documents/Thesis/data/askubuntu/"
          "vector/vectors_pruned.200.txt.gz --save_dir {}/check_runs1 --max_epoch 1 > gianadoume1.txt".format(code_dir))

os.system("python tags_prediction/main.py --corpus_w_tags /home/christina/Documents/Thesis/data/askubuntu/additional/"
          "texts_raw_with_tags_str.txt --df_path /home/christina/Documents/Thesis/data/askubuntu/"
          "additional/data_frame_corpus_str.csv --tags_file /home/christina/Documents/Thesis/data/askubuntu/"
          "additional/tags_files/valid_train_tags.p --embeddings /home/christina/Documents/Thesis/data/askubuntu/"
          "vector/vectors_pruned.200.txt.gz --save_dir {}/check_runs2 --max_epoch 1 > gianadoume2.txt".format(code_dir))
