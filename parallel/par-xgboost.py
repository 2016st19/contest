# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

loan_data_train = pd.read_csv('train/browse_history_train.txt',
                                       header = None)
loan_data_test = pd.read_csv('test/browse_history_test.txt',
                                       header = None)
col_names = ['userid', 'tm_encode_3', 'bank_id', 'prior_account', 'prior_repay',
             'credit_limit', 'account_balance', 'minimun_repay', 'consume_count',
             'account', 'adjust_account', 'circulated_interest', 'avaliable_balance',
             'cash_limit', 'repay_state', 'browse_data', 'browse_tag', 'sex', 
             'occupation', 'education', 'marriage', 'household', 'loan_time']
loan_data_train.columns = col_names
loan_data_test.columns = col_names
loan_data = pd.concat([loan_data_train, loan_data_test])
loan_data.index = loan_data['userid']
print (loan_data.head(5))

loan_data.drop('userid',
                axis = 1,
                inplace = True)

# 构造新特征
loan_data['time'] = loan_data['loan_time'] - loan_data['tm_encode_3']
#loan_data['f2'] = loan_data['income_tm'] + loan_data['outcome_tm'] - 2 * loan_data['loan_time']
loan_data['f3'] = loan_data['avaliable_balance'] - loan_data['prior_account'] - loan_data['adjust_account'] - loan_data['account']
loan_data['f4'] = loan_data['browse_data'] - loan_data['avaliable_balance']
#oan_data['f4'] = loan_data['browse_data'] - loan_data['minimun_repay']

# 对性别、职业等因子变量，构造其哑变量
category_col = ['sex', 'occupation', 'education', 'marriage', 'household', 'repay_state']
def set_dummies(data, colname):
    for col in colname:
        data[col] = data[col].astype('category')
        dummy = pd.get_dummies(data[col])
        dummy = dummy.add_prefix('{}#'.format(col))
        data.drop(col,
                  axis = 1,
                  inplace = True)
        data = data.join(dummy)
    return data
loan_data = set_dummies(loan_data, category_col)

# overdue_train，这是我们模型所要拟合的目标
target = pd.read_csv('train/overdue_train.txt',
                         header = None)
target.columns = ['userid', 'label']
target.index = target['userid']
target.drop('userid',
            axis = 1,
            inplace = True)
# 构建模型
# 分开训练集、测试集
train = loan_data.iloc[0: 55596, :]
test = loan_data.iloc[55596:, :]
# 避免过拟合，采用交叉验证，验证集占训练集20%，固定随机种子（random_state)
train_X, test_X, train_y, test_y = train_test_split(train,
                                                    target,
                                                    test_size = 0.05,
                                                    random_state = 0)
train_y = train_y['label']
test_y = test_y['label']

import xgboost as xgb
dtrain = xgb.DMatrix(train_X, label=train_y)
dtest = xgb.DMatrix(test_X, label=test_y)
dtest_test = xgb.DMatrix(test)

params = {
            'booster':'gbtree',
            'objective':'binary:logistic',
            'eta':0.1,
            'max_depth':10,
            'subsample':1.0,
            'min_child_weight':5,
            'colsample_bytree':0.2,
            'scale_pos_weight':0.1,
            'eval_metric':'auc',
            'gamma':0.2,            
            'lambda':300
}
watchlist = [(dtrain,'train'),(dtest,'val')]
model = xgb.train(params,dtrain,num_boost_round=1000,evals = watchlist)#100000

pred = model.predict(dtest_test)
result = pd.DataFrame(pred)
result.index = test.index
result.columns = ['probability']
print (result.head(5))
# 输出结果
result.to_csv('result.csv')
