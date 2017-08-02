from flask import Flask
import json
import argparse
import sys
from utils import read_questions, read_results_rows, questions_index
import random

app = Flask(__name__)

results = None
q_idx = None


@app.route("/show_performance_less/<value>")
def show_performance_less(value):
    random.shuffle(results)
    to_show = {}
    for i, (qid, qidtrue, qidranked, rankedscores, rankedlabels, map_, mrr_, pat1_, pat5_) in enumerate(results):
        if sum([map_, mrr_, pat1_, pat5_])/4. < float(value):
            predicted = []
            for j, x in enumerate(qidranked):
                if x in qidtrue:
                    assert rankedlabels[j] == 1, 'opa'
                    predicted.append('%s\t%f' % (q_idx[x].upper(), rankedscores[j]))
                else:
                    assert rankedlabels[j] == 0, 'opa'
                    predicted.append('%s\t%f' % (q_idx[x].lower(), rankedscores[j]))
            if len(to_show.keys()) > 20:
                break
            to_show[qid] = {
                'q': q_idx[qid],
                'predicted': predicted
            }
    return json.dumps(to_show, encoding='utf8', indent=2)


@app.route("/show_performance_more/<value>")
def show_performance_more(value):
    random.shuffle(results)
    to_show = {}
    for i, (qid, qidtrue, qidranked, rankedscores, rankedlabels, map_, mrr_, pat1_, pat5_) in enumerate(results):
        if sum([map_, mrr_, pat1_, pat5_])/4. > float(value):
            predicted = []
            for j, x in enumerate(qidranked):
                if x in qidtrue:
                    assert rankedlabels[j] == 1, 'opa'
                    predicted.append('%s\t%f' % (q_idx[x].upper(), rankedscores[j]))
                else:
                    assert rankedlabels[j] == 0, 'opa'
                    predicted.append('%s\t%f' % (q_idx[x].lower(), rankedscores[j]))
            if len(to_show.keys()) > 20:
                break
            to_show[qid] = {
                'q': q_idx[qid][0],
                'predicted': predicted
            }
    return json.dumps(to_show, encoding='utf8', indent=2)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--corpus_file", type=str, default="")
    argparser.add_argument("--results_file", type=str, default="")
    args = argparser.parse_args()

    results = list(read_results_rows(args.results_file))
    questions = list(read_questions(args.corpus_file))
    q_idx = questions_index(questions)

    app.run()
