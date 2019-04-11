import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from util import *
from parameters import *

import warnings
warnings.filterwarnings('ignore')

output_result = "output_result.md"


def kfoldAndTest(train_data, train_label, test_data, test_label, k=5):

    # ============== Training ============== #

    # Generate report (also in addLabels.py)
    with open(output_result, 'w') as report:
        report.write('# Report of the SemEval Task 7 #\n\n')

    kf = KFold(n_splits=k)

    print("Doing %d-fold on training set..." % k)

    collected_result_list = []

    p, r, f1 = 0.0, 0.0, 0.0

    # ============== K-fold ============== #

    for i, (train_index, valid_index) in enumerate(kf.split(train_data)):
        X_train, X_valid = train_data[train_index], train_data[valid_index]
        y_train, y_valid = train_label[train_index], train_label[valid_index]
        pred = classify(X_train, y_train, X_valid)

        print('The %dth round result:' % (i+1))
        tp, tr, tf1, result_list = scoreSelf(pred, y_valid)
        print('P: {:.2f}, R: {:.2f}, F1: {:.2f}'.format(
            tp*100, tr*100, tf1*100))
        p += tp
        r += tr
        f1 += tf1
        collected_result_list.append(result_list)

    p /= k
    r /= k
    f1 /= k
    avf1 = (2 * p * r) / (p + r) if (p + r) else 0

    # K-fold validation
    with open(output_result, 'a') as report:
        report.write('## Average %d-fold validation (training) ##\n\n' % k)
        report.write('Train F1|Train P|Train R\n')
        report.write('--------|-------|-------\n')
        report.write(' %.2f | %.2f | %.2f\n\n' % (f1*100, p*100, r*100))

    print('results for {}-fold validation: F1 = {:.4f}%'.format(k, f1))
    print('P = {:.4f}%, R = {:.4f}%, Aver F1 = {:.4f}%'.format(p, r, avf1))

    # Average of Each Relation
    with open(output_result, 'a') as report:
        report.write('## Average f1 of Each Relation ##\n\n')
        report.write('USAGE|TOPIC|RESULT|PART_WHOLE|MODEL-FEATURE|COMPARE\n')
        report.write('-----|-----|------|----------|-------------|-------\n')

    tf_list = []

    for relaid in range(len(rela2id.keys())):
        temp_list = []
        for fold in range(k):
            temp_list.append(collected_result_list[fold][relaid])

        df = pd.DataFrame(temp_list)
        tp, tr, tf = df.mean()
        avf1 = (2 * tp * tr) / (tp + tr) if (tp + tr) else 0
        print('{} P = {:.4f}%, R = {:.4f}%, Aver F1 = {:.4f}%, f1 = {:.4f}%'.format(
            id2rela[relaid], tp, tr, avf1, tf))

        string = '%.2f' % (tf*100)
        tf_list.append(string)

    with open(output_result, 'a') as report:
        report.write(' | '.join(tf_list))
        report.write('\n\n')

    # ============== Test ============== #

    print("Testing on test set...")
    prediction = classify(train_data, train_label, test_data)
    with open('data/1.1.test.relations.txt', 'r') as f:
        test = [ii.strip() for ii in f.readlines()]

    # result = [id2rela[ii] for ii in prediction]
    # result = [result[ii] + test[ii] for ii in range(len(result))]
    # with open('result.txt', 'w') as f:
    #     f.write('\n'.join(result))

    p, r, f1, _ = scoreSelf(prediction, test_label)

    # Generate report (also in average.py)
    # K-fold validation
    with open(output_result, 'a') as report:
        report.write('## Average of test set ##\n\n')
        report.write('Test F1|Test P|Test R\n')
        report.write('-------|-----|-------\n')
        report.write(' %.2f | %.2f | %.2f\n' %
                     (f1*100, p*100, r*100))


def classify(train_data, train_label, test_data):
    # clf = SVC()
    # clf = LinearSVC()
    clf = LogisticRegression()
    # clf = DecisionTreeClassifier()

    clf.fit(train_data, train_label)
    return clf.predict(test_data)


def loadDataset(filename):
    pickle_matrix, pickle_label = load_bigger(filename)
    return pickle_matrix, pickle_label


if __name__ == "__main__":
    train_data, train_label = loadDataset('%strain_data.pkl' % models_dir)
    test_data, test_label = loadDataset('%stest_data.pkl' % models_dir)
    kfoldAndTest(train_data, train_label, test_data, test_label, k=5)
