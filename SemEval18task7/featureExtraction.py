import itertools
from parameters import *
import sys
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import re
from typing import Tuple, List, Dict


def create_relation_index(rel_path: str) -> Dict:
    # @return dict key: title_name value: (e1, e2): [relation, reverse]
    # @example 'H01-1001': {('H01-1001.5', 'H01-1001.7'): ['USAGE', True]}
    rel_idx = {}
    typ_pattern = r'[^(]*'
    rel_pattern = r'\((.*?)\)'

    with open(rel_path) as f:
        for line in f:
            rel = re.findall(rel_pattern, line)[0]
            typ = re.findall(typ_pattern, line)[0]
            split_rel = rel.split(',')
            abs_id = split_rel[0].split('.')[0]
            if abs_id not in rel_idx:
                rel_idx[abs_id] = {}
            e1_e2 = tuple(split_rel[:2])
            len_split_rel = len(split_rel)
            if len_split_rel == 3:
                rel_idx[abs_id][e1_e2] = [typ, True]
            elif len_split_rel == 2:
                rel_idx[abs_id][e1_e2] = [typ, False]

    return rel_idx


def collect_texts(dat_file: str) -> Tuple[dict, dict]:
    txt_idx, ent_idx = {}, {}
    tree = ET.parse(dat_file)
    doc = tree.getroot()

    for txt in doc:  # looping over each abstract in entire xml doc
        abs_id = txt.get('id')
        whole_abs_text = ''
        for child in txt:  # children are title and abstract, H93-1076 has entities in title, but no relation
            for el in child.iter():
                tag = el.tag
                if tag == 'title':
                    continue
                elif tag == 'abstract':
                    abs_text = el.text
                    if abs_text:
                        whole_abs_text += abs_text
                elif tag == 'entity':
                    ent_id = el.get('id')
                    # ent_text = el.text
                    ent_text = ''.join(e for e in el.itertext() if e)
                    # collect id to entity mapping to be used later
                    ent_idx[ent_id] = ent_text
                    ent_tail = el.tail
                    if ent_tail:
                        if ent_tail[0] == ' ':
                            whole_abs_text += ent_id + ent_tail
                        else:
                            whole_abs_text += ent_id + ' ' + ent_tail
                    else:
                        whole_abs_text += ent_id + ' '
        txt_idx[abs_id] = whole_abs_text

    return txt_idx, ent_idx


def create_record(txts, rel_idx: dict, ent_idx: dict) -> List[tuple]:
    """
    @return list(tuple) (title, [sentents], e1, e2, label)
    @example('H01-1001', ['information_retrieval_techniques','use','a','histogram','of','keywords'],'H01-1001.5','H01-1001.7','USAGE REVERSE',6)
    """
    recs = []

    for abs_id, rels in rel_idx.items():
        for rel, info in rels.items():
            e1, e2 = rel
            rel_patt = e1 + r'(.*?)' + e2

            rel_text_between = re.findall(
                rel_patt, txts[abs_id].replace('\n', ''))[0]

            rel_text_full = e1 + rel_text_between + e2
            tokens = rel_text_full.split()
            # replace entity ids with actual entities
            for i, token in enumerate(tokens):
                if token in ent_idx:
                    # if entity in relation, join with underscores if multi-word
                    if i == 0 or i == len(tokens)-1:
                        tokens[i] = '_'.join(
                            toke for toke in ent_idx[token].split())
                    else:
                        tokens[i] = ent_idx[token]
            tokens_with_punc = list(merge_punc(tokens))
            s_len = len(tokens_with_punc)
            # typ = info[0] + ' REVERSE' if info[1] else info[0]
            typ = info[0]
            recs.append(tuple([abs_id, tokens_with_punc, e1, e2, typ, s_len]))

    return recs


# Generator
def merge_punc(tkn_lst: list):
    to_merge = {',', '.', ':', ';'}
    seq = iter(tkn_lst)
    curr = next(seq)
    for nxt in seq:
        if nxt in to_merge:
            curr += nxt
        else:
            yield curr
            curr = nxt
    yield curr


# Without sorted
def write_record(rec_out: str, recs: list):
    # used for writing internal python containers to file
    with open(rec_out, 'w+') as out:
        out.write('\n'.join([str(ii) for ii in recs]))


# Sorted
def index_uniques(uni_out: str, uniqs: list):
    # used for writing unique info already in string format to file
    with open(uni_out, 'w+') as voc:
        voc.write('\n'.join([str(ii) for ii in sorted(uniqs)]))


if __name__ == '__main__':
    folds = int(sys.argv[1])  # flod num for cross validation
    # input files
    path_to_relations = data_dir + task_number + '.relations.txt'
    path_to_test_relations = data_dir + 'keys.test.' + task_number + '.txt'
    path_to_train_data = data_dir + task_number + '.text.xml'
    path_to_test_data = data_dir + task_number + '.test.text.xml'
    # output files
    train_record_file = features_dir + 'record_train.txt'
    test_record_file = features_dir + 'record_test.txt'
    vocab_output = features_dir + 'vocab.txt'
    shapes_output = features_dir + 'shapes.txt'
    e1_context_output = features_dir + 'e1_context.txt'
    e2_context_output = features_dir + 'e2_context.txt'
    label_output = features_dir + 'labels.txt'

    training_relation_index = create_relation_index(path_to_relations)
    test_relation_index = create_relation_index(path_to_test_relations)
    training_text_index, training_entity_index = collect_texts(
        path_to_train_data)
    test_text_index, test_entity_index = collect_texts(path_to_test_data)
    training_records = create_record(
        training_text_index, training_relation_index, training_entity_index)
    unique_test_words, unique_test_context_e1, unique_test_context_e2 = set(), set(), set()

    if folds == 0:
        # competition run
        test_records = create_record(
            test_text_index, test_relation_index, test_entity_index)
        write_record(train_record_file, training_records)
        write_record(test_record_file, test_records)
        unique_test_words = set(
            [word for rec in test_records for word in rec[1]])
        unique_test_context_e1 = set([r[1][1]
                                      for r in test_records if len(r) >= 3])
        unique_test_context_e2 = set([r[1][-2]
                                      for r in test_records if len(r) >= 3])

    else:
        # cross-val development
        test_size = len(training_records) / folds
        for k in range(1, folds + 1):
            test_start = int(test_size * (k-1))
            test_end = int(test_size * k)
            cv_test_split = training_records[test_start:test_end]
            cv_train_split = training_records[:test_start] + \
                training_records[test_end:]
            cv_test_output = features_dir + 'record_test' + str(k) + '.txt'
            cv_train_output = features_dir + 'record_train' + str(k) + '.txt'
            write_record(cv_test_output, cv_test_split)
            write_record(cv_train_output, cv_train_split)

    # collect unique words and entity context from training data
    unique_training_words = set(
        [word for rec in training_records for word in rec[1]])
    unique_training_context_e1 = set(
        [r[1][1] for r in training_records if len(r) >= 3])
    unique_training_context_e2 = set(
        [r[1][-2] for r in training_records if len(r) >= 3])
    # add unique words and entity context from test data
    unique_words = unique_training_words.union(unique_test_words)
    unique_context_e1 = unique_training_context_e1.union(
        unique_test_context_e1)
    unique_context_e2 = unique_training_context_e2.union(
        unique_test_context_e2)
    unique_labels = {r[4] for r in training_records}
    num_shape_dims = 7  # change according to number of shape features
    unique_shapes = list(itertools.product(range(2), repeat=num_shape_dims))

    # load data to file
    index_uniques(vocab_output, unique_words)            # features/vocab.txt
    index_uniques(label_output, unique_labels)           # features/labels.txt
    index_uniques(e1_context_output, unique_context_e1)  # follow entity1
    index_uniques(e2_context_output, unique_context_e2)  # before entity2
    write_record(shapes_output, unique_shapes)           # shape(0, 0, 0, 0)
