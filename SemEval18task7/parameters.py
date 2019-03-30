features_dir = 'feature/'
models_dir = 'model/'
data_dir = 'data/'
result_dir = 'results/'
fire_words = True
fire_shapes = False
fire_embeddings = True
fire_clusters = False
fire_e1_context = True
fire_e2_context = True
before_e2 = False  # defaults to padding after e1 if set to False
task_number = '1.1'

rela2id = {
    'USAGE': 0,
    'TOPIC': 1,
    'RESULT': 2,
    'PART_WHOLE': 3,
    'MODEL-FEATURE': 4,
    'COMPARE': 5
}

id2rela = {
    0: 'USAGE',
    1: 'TOPIC',
    2: 'RESULT',
    3: 'PART_WHOLE',
    4: 'MODEL-FEATURE',  # MODEL
    5: 'COMPARE'        # COMPARISON
}
