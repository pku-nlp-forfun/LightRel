'''
@Author: gunjianpan
@Date:   2019-03-29 14:33:41
@Last Modified by:   gunjianpan
@Last Modified time: 2019-03-30 14:43:32
'''

import os
import re
import pandas as pd

from parameters import *
from util import *

p, r, f1 = 0.0, 0.0, 0.0
result = []

results_files = [i for i in os.listdir(result_dir)]
k = len(results_files)
for file in results_files:
    tp, tr, tf1, tre = load_bigger(result_dir + file)
    p += tp
    r += tr
    f1 += tf1
    result.append(tre)

p /= k
r /= k
f1 /= k
avf1 = (2 * p * r) / (p + r) if (p + r) else 0
print('results for {} -fold validation: F1 = {:.4f}%'.format(k, f1))
print('P = {:.4f}%, R = {:.4f}%, Aver F1 = {:.4f}%'.format(p, r, avf1))

# Generate report (also in addLabels.py)
with open('result.md', 'w') as report:
    report.write('# Report of the SemEval Task 7 #\n\n')

# K-fold validation
with open('result.md', 'a') as report:
    report.write('## Average %d-fold validation (training) ##\n\n' % k)
    report.write('Train F1|Train P|Train R\n')
    report.write('--------|-------|-------\n')
    report.write(' %.2f | %.2f | %.2f\n\n' % (f1*100, p*100, r*100))

# Average of Each Relation
with open('result.md', 'a') as report:
    report.write('## Average f1 of Each Relation ##\n\n')
    report.write('USAGE|TOPIC|RESULT|PART_WHOLE|MODEL-FEATURE|COMPARE\n')
    report.write('-----|-----|------|----------|-------------|-------\n')

tf_list = []

for ii in range(len(rela2id.keys())):
    temp_list = []
    for jj in range(k):
        temp_list.append(result[jj][ii])
    df = pd.DataFrame(temp_list)
    tp, tr, tf = df.mean()
    avf1 = (2 * tp * tr) / (tp + tr) if (tp + tr) else 0
    print('{} P = {:.4f}%, R = {:.4f}%, Aver F1 = {:.4f}%, f1 = {:.4f}%'.format(
        id2rela[ii], tp, tr, avf1, tf))

    string = '%.2f' % (tf*100)
    tf_list.append(string)

with open('result.md', 'a') as report:
    report.write(' | '.join(tf_list))
    report.write('\n\n')
