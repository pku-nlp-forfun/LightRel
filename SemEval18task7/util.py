'''
@Author: gunjianpan
@Date:   2019-03-29 19:05:54
@Last Modified by:   gunjianpan
@Last Modified time: 2019-04-10 20:32:46
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
def fastF1(result, predict):
    ''' f1 score '''
    true_total, r_total, p_total, p, r = 0, 0, 0, 0, 0
    total_list = []
    for trueValue in range(6):
        trueNum, recallNum, precisionNum = 0, 0, 0
        for index, values in enumerate(result):
            if values == trueValue:
                recallNum += 1
                if values == predict[index]:
                    trueNum += 1
            if predict[index] == trueValue:
                precisionNum += 1
        R = trueNum / recallNum if recallNum else 0
        P = trueNum / precisionNum if precisionNum else 0
        true_total += trueNum
        r_total += recallNum
        p_total += precisionNum
        p += P
        r += R
        f1 = (2 * P * R) / (P + R) if (P + R) else 0
        print(id2rela[trueValue], P, R, f1)
        total_list.append([P, R, f1])
    p /= 6
    r /= 6
    micro_r = true_total / r_total
    micro_p = true_total / p_total
    macro_f1 = (2 * p * r) / (p + r) if (p + r) else 0
    micro_f1 = (2 * micro_p * micro_r) / (micro_p +
                                          micro_r) if (micro_p + micro_r) else 0
    print('P: {:.2f}%, R: {:.2f}%, Micro_f1: {:.2f}%, Macro_f1: {:.2f}%'.format(
        p*100, r*100, micro_f1 * 100, macro_f1*100))
    return p, r, macro_f1, total_list


def scoreSelf(predict, result=None):
    if result is None:
        with open('{}keys.test.{}.txt'.format(data_dir, task_number), 'r') as f:
            result = [rela2id[ii.split('(')[0]] for ii in f.readlines()]
    return fastF1(result, predict)
