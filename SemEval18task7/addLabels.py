import ast
import sys

from parameters import *
from util import *


def read_record(file):
    with open(file) as f:
        return [ast.literal_eval(line) for line in f]


if __name__ == '__main__':
    which_set = '' if sys.argv[1] == '0' else sys.argv[1]
    path_to_predictions = models_dir + 'predictions.txt'
    path_to_test = features_dir + 'record_test' + which_set + '.txt'
    test_record = read_record(path_to_test)
    # list of entities in order of test case
    relation_list = [(rec[2:5]) for rec in test_record]

    result = [rela2id[ii[-1]] for ii in relation_list]
    with open(path_to_predictions, 'r') as predicts:
        prediction = [int(ii.strip()) for ii in predicts.readlines()]
    p, r, f1, result_t = scoreSelf(prediction, result)
    dump_bigger([p, r, f1, result_t],
                '{}f1{}.pkl'.format(result_dir, which_set))

    if which_set == '':
        print('P = {:.4f}%, R = {:.4f}%, Aver F1 = {:.4f}%'.format(p, r, f1))
        for ii, jj in enumerate(result_t):
            tp, tr, tf1 = jj
            print('{} P = {:.4f}%, R = {:.4f}%, Aver F1 = {:.4f}%'.format(
                id2rela[ii], tp, tr, tf1))

        # Generate report (also in average.py)
        # K-fold validation
        with open('result.md', 'a') as report:
            report.write('## Average of test set ##\n\n')
            report.write('Test F1|Test P|Test R\n')
            report.write('-------|-----|-------\n')
            report.write(' %.2f | %.2f | %.2f\n' %
                         (f1*100, p*100, r*100))
