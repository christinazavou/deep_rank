# -*- coding: utf-8 -*-
import os
import argparse
import sys

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--corpus", type=str)
    argparser.add_argument("--train", type=str, default="")
    argparser.add_argument("--test", type=str, default="")
    argparser.add_argument("--dev", type=str, default="")
    argparser.add_argument("--out_dir", type=str, default="")

    this_dir = os.path.dirname(os.path.realpath(__file__))
    main_file = os.path.join(this_dir, "qa", "main.py")

    run_cmd = "python \"{}\" \"{}\" \"{}\" \"{}\" \"{}\"".format(main_file, )
    os.system(run_cmd)
