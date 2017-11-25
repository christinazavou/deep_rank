
import pickle
from utils import str2int_list, read_df
import numpy as np
import argparse
import sys
from utils import read_eval_rows


room_for_direct_indirect_improvement = list(
    {180432, 343901, 25746, 314644, 290926, 209891, 214355, 42281, 382782, 14168, 171521, 397771, 20015, 485989,
     352815, 42284, 315967, 280724, 497191, 423702, 163982, 103456, 74690, 448135, 20015} |
    {499786, 183527, 448135, 195507, 378661, 418357, 452297, 101650, 52413, 224609, 3678, 457008, 103453, 173299,
     277976, 401490, 165662, 103466, 115432, 277976, 128732, 3659, 294551, 378661, 234851, 520011}
)

room_for_no_misc_improvement = list(
    {422452, 128732, 491861, 367963, 324818, 68888, 293157, 138950, 77787, 285609, 69201, 119803, 393400, 406357,
     469945, 48331, 174593, 409418, 132745, 501831, 406329, 33378, 268194, 195543, 146530, 96821, 405072, 81824,
     34670, 22667, 250954, 370758, 422452, 491861, 437495, 481638, 138950, 184840} |
    {5440, 8508, 18531, 26150, 37656, 59922, 73376, 91119, 103464, 119817, 155491, 155494, 294471, 388829, 96433,
     429043, 485968, 520900}
)


argparser = argparse.ArgumentParser(sys.argv[0])
argparser.add_argument("--qr_ids", type=str, default="")
argparser.add_argument("--tp_ids", type=str, default="")
argparser.add_argument("--test_file", type=str, default="")
args = argparser.parse_args()


tp_ids = pickle.load(open(args.tp_ids, 'rb'))
qr_ids = pickle.load(open(args.qr_ids, 'rb'))
print len(qr_ids['bad_queries'])
print len(qr_ids['best_queries'])


for q_id, q_ids_similar, q_ids_candidates in read_eval_rows(args.test_file):
    if q_id in qr_ids['bad_queries']:
        qr_ids['bad_queries'].extend(str2int_list(q_ids_candidates))
    elif q_id in qr_ids['best_queries']:
        qr_ids['best_queries'].extend(str2int_list(q_ids_candidates))
print len(qr_ids['bad_queries'])
print len(qr_ids['best_queries'])


qr_bad_in_tp_bad, babac = [], 0
qr_bad_in_tp_best, babec = [], []
qr_best_in_tp_bad, bebac = [], []
qr_best_in_tp_best, bebec = [], 0
for qid in qr_ids['bad_queries']:
    if qid in tp_ids['bad_queries']:
        qr_bad_in_tp_bad.append(qid)
    elif qid in tp_ids['best_queries']:
        qr_bad_in_tp_best.append(qid)

for qid in qr_ids['best_queries']:
    if qid in tp_ids['bad_queries']:
        qr_best_in_tp_bad.append(qid)
    elif qid in tp_ids['best_queries']:
        qr_best_in_tp_best.append(qid)


for q_id, q_ids_similar, q_ids_candidates in read_eval_rows(args.test_file):
    if q_id in qr_bad_in_tp_bad:
        babac += 1
    elif q_id in qr_bad_in_tp_best:
        babec.append(q_id)
    elif q_id in qr_best_in_tp_bad:
        bebac.append(q_id)
    elif q_id in qr_best_in_tp_best:
        bebec += 1

print 'qr bad tp bad: ', len(qr_bad_in_tp_bad), babac  # 944 44
print 'qr bad tp best: ', len(qr_bad_in_tp_best), len(babec)  # 827 44
print 'qr best tp bad: ', len(qr_best_in_tp_bad), len(bebac)  # 622 35
print 'qr best tp best: ', len(qr_best_in_tp_best), bebec  # 486 16

print 'babec: ', babec
print 'bebac: ', bebac

room_for_improve_in_bad_bad = []
room_for_improve_in_bad_best = []
for q_id in room_for_direct_indirect_improvement:
    if q_id in qr_bad_in_tp_bad:
        room_for_improve_in_bad_bad.append(q_id)
    elif q_id in qr_best_in_tp_best:
        room_for_improve_in_bad_best.append(q_id)

no_room_for_improve_in_bad_bad = []
no_room_for_improve_in_bad_best = []
for q_id in room_for_no_misc_improvement:
    if q_id in qr_bad_in_tp_bad:
        no_room_for_improve_in_bad_bad.append(q_id)
    elif q_id in qr_best_in_tp_best:
        no_room_for_improve_in_bad_best.append(q_id)

print 'room for improve in bad: ', room_for_improve_in_bad_bad
print 'room for improve in best: ', room_for_improve_in_bad_best
print 'no room for improve in bad: ', no_room_for_improve_in_bad_bad
print 'no room for improve in best: ', no_room_for_improve_in_bad_best
