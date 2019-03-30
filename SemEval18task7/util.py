'''
@Author: gunjianpan
@Date:   2019-03-29 19:05:54
@Last Modified by:   gunjianpan
@Last Modified time: 2019-03-30 22:37:48
'''
import os
import pickle

from numba import jit
from parameters import *


def dump_bigger(data, output_file):
    """
    pickle.dump big file which size more than 4GB
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(data, protocol=4)
    with open(output_file, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def load_bigger(input_file):
    """
    pickle.load big file which size more than 4GB
    """
    max_bytes = 2**31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(input_file)
    with open(input_file, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    return pickle.loads(bytes_in)


@jit
def fastF1(result, predict, trueValue):
    """
    f1 score
    """
    trueNum = 0
    recallNum = 0
    precisionNum = 0
    for index, values in enumerate(result):
        if values == trueValue:
            recallNum += 1
            if values == predict[index]:
                trueNum += 1
        if predict[index] == trueValue:
            precisionNum += 1
    R = trueNum / recallNum if recallNum else 0
    P = trueNum / precisionNum if precisionNum else 0
    f1 = (2 * P * R) / (P + R) if (P + R) else 0
    # print(id2rela[trueValue], P, R, f1)
    return P, R, f1


def scoreSelf(predict, result=None):
    if result is None:
        with open('%skeys.test.%s.txt' % (data_dir, task_number), 'r') as f:
            result = [rela2id[ii.split('(')[0]] for ii in f.readlines()]
    p, r = 0, 0
    result_list = []
    for ii in range(6):
        tp, tr, tf1 = fastF1(result, predict, ii)
        result_list.append([tp, tr, tf1])
        p += tp
        r += tr
    p /= 6
    r /= 6
    f1 = (2 * p * r) / (p + r) if (p + r) else 0
    # print('Once', p, r, f1)
    return p, r, f1, result_list
