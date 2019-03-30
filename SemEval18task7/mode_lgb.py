'''
@Author: gunjianpan
@Date:   2019-03-30 17:28:24
@Last Modified by:   gunjianpan
@Last Modified time: 2019-03-30 21:05:44
'''

import lightgbm as lgb
import numpy as np
import pandas as pd
import warnings
import threading
import time

from datetime import datetime
from numba import jit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from util import *
from parameters import *

warnings.filterwarnings('ignore')

data_path = 'model/lightgbm/data/'
model_path = 'model/lightgbm/model/'
pickle_path = 'model/lightgbm/pickle/'
prediction_path = 'model/lightgbm/prediction/'
v = '2'
# t = '_total'
t = ''
start = []


class semEval(object):
    """
    semEval 2018 Task7 for lightGbm
    """

    def __init__(self, do_pre=False):
        self.id2file = {}
        self.id2lab = {}
        self.detail_map = {}
        self.detail_pickle = {}
        self.f1_max_index = 0.5
        self.f1_map = {index: 0 for index in range(0, 5)}
        self.version = datetime.now().strftime("%m%d%H%M")
        self.seed = 333
        self.EARLY_STOP = 300
        self.OPT_ROUNDS = 2444
        self.MAX_ROUNDS = 300000
        self.evaluate_num = 0

        self.params = {
            'boosting': 'gbdt',
            'objective': 'multiclass',
            "num_class": 6,
            'learning_rate': 0.01,
            'max_depth': -1,
            'min_child_samples': 20,
            'max_bin': 255,
            'subsample': 0.85,
            'subsample_freq': 10,
            'colsample_bytree': 0.8,
            'min_child_weight': 0.001,
            'subsample_for_bin': 200000,
            'min_split_gain': 0,
            'reg_alpha': 0,
            'reg_lambda': 0,
            'num_leaves': 63,
            'seed': self.seed,
            'nthread': 20,
            'metric': "None",
            "verbose": -1
        }

    def evaluate_f1(self, preds, train_data):

        labels = train_data.get_label()
        preds = [int(ii + 0.5) for ii in preds]

        _, _, f1, _ = scoreSelf(preds, labels)
        return 'f1', f1, True

    def pre_data(self, pre, slices):
        """
        prepare data
        """
        if slices is None:
            return pre
        else:
            wait_columns = pre.columns
            wait = pd.DataFrame(pre, columns=wait_columns)
            return wait

    def pre_train(self):
        print('<<<<load data begin<<<<')
        pre = load_bigger('%strain_data.pkl' % models_dir)
        test_pre = load_bigger('%stest_data.pkl' % models_dir)
        df_pre = pd.DataFrame(pre[0])
        test_df_pre = pd.DataFrame(test_pre[0])
        df_max = df_pre.max()
        df_min = df_pre.min()
        df_del = [ii for ii in range(len(df_max))]
        df_columns_other = [ii for ii in range(
            len(df_max)) if ii not in df_del]

        self.X_test = pd.DataFrame(test_df_pre, columns=df_columns_other)
        self.X_train = pd.DataFrame(df_pre, columns=df_columns_other)
        self.y_test = test_pre[1]
        self.y_train = pre[1]
        print('>>>>load data over>>>>')

    def train_model(self):
        """
        train model by lightgbm
        """

        print('Start training...')

        categorical = []

        dtrain = lgb.Dataset(self.X_train,
                             label=self.y_train,
                             feature_name=list(self.X_train.columns),
                             categorical_feature=categorical)

        model = lgb.train(self.params,
                          dtrain,
                          num_boost_round=self.OPT_ROUNDS,
                          valid_sets=[dtrain],
                          valid_names=['train'],
                          verbose_eval=100,
                          feval=self.evaluate_f1)

        importances = pd.DataFrame({'features': model.feature_name(),
                                    'importances': model.feature_importance()})

        importances.sort_values('importances', ascending=False, inplace=True)

        model.save_model('{}{}.model'.format(model_path, self.version))
        importances.to_csv(
            '{}{}_importances.csv'.format(model_path, self.version), index=False)

        self.gbm = model
        self.dtrain = dtrain

    def evaulate_model(self, model=True, slices=None):
        """
        evaulate model by lightgbm
        """
        print('Start predicting...')

        y_pred = self.gbm.predict(
            self.X_test, num_iteration=self.gbm.best_iteration)

        scoreSelf(y_pred, self.y_test)

    def optimize_model(self, index=None):
        """
        optimize model by lightgbm
        """
        # print(self.X_train.columns, len(self.X_train.columns), self.y_train[0])
        dtrain = lgb.Dataset(self.X_train,
                             label=self.y_train,
                             feature_name=list(self.X_train.columns),
                             categorical_feature=[])

        eval_hist = lgb.cv(self.params,
                           dtrain,
                           nfold=5,
                           num_boost_round=self.MAX_ROUNDS,
                           early_stopping_rounds=self.EARLY_STOP,
                           verbose_eval=50,
                           seed=self.seed,
                           shuffle=True,
                           feval=self.evaluate_f1,
                           metrics="None"
                           )
        result = [self.version]
        result.append('best n_estimators:' + str(len(eval_hist['f1-mean'])))
        result.append('best cv score:' + str(eval_hist['f1-mean'][-1]) + '\n')
        with open(model_path + v + 'result', 'a') as f:
            f.write('\n'.join([str(index) for index in result]))
        print('best n_estimators:', len(eval_hist['f1-mean']))
        print('best cv score:', eval_hist['f1-mean'][-1])
        self.OPT_ROUNDS = len(eval_hist['f1-mean'])


if __name__ == '__main__':
    semEval = semEval()
    semEval.pre_train()
    semEval.optimize_model()
    semEval.train_model()
    semEval.evaulate_model()
