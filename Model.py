import pandas as pd
import numpy as np
import warnings
import time
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

warnings.filterwarnings('ignore')
path = ''
train_num = 20000
test_num = 1129000

def pre_process(fill=False):
    df_all = pd.read_csv(path + 'features.csv', encoding='gbk',parse_dates=['due_date','auditing_date_x'],index_col=0)
    train_y = pd.DataFrame()
    train_y['label'] = df_all['label'][:train_num].values
    train_y['repay_amt'] = df_all['repay_amt'][:train_num].values
    train_y['due_amt'] = df_all['due_amt'][:train_num].values
    test_due_amt = df_all[['due_amt']][:train_num]
    sub = df_all[['listing_id', 'auditing_date_x', 'due_amt', 'due_date']][test_num:]

    # 去掉无关特征
    drop_cols = ['auditing_date_x', 'due_date', 'repay_date', 'last_list_time', 'reg_mon', 'info_insert_date',
                 'repay_amt']
    df_all.drop(columns=drop_cols, axis=1, inplace=True)

    # 将分类标签转换编码
    cate_cols = ['day_term', 'gender', 'cell_province', 'id_province', 'new_tag', 'same_province_label', 'map_age',
                 'id_city']
    df_all = pd.get_dummies(df_all, columns=cate_cols)

    # 进行数据标准化
    scaler = preprocessing.MaxAbsScaler()
    scale_cols = ['advance_mean', 'advance_std', 'due_amt', 'mean_principal',
                  'recent_list_num', 'std_principal', 'total_list_count', 'term',
                  'rate_x', 'principal', 'time_difference', 'delta_principal',
                  'age', 'due_month', 'due_day', 'due_dayofweek', 'aud_dayofweek']
    df_all[scale_cols] = pd.DataFrame(scaler.fit_transform(df_all[scale_cols]), columns=scale_cols)

    # 用word_to_voc提取用户标签信息
    df_all['taglist'] = df_all['taglist_x'].astype('str').apply(lambda x: x.strip().replace('|', ' ').strip())
    tag_cv = CountVectorizer(min_df=10, max_df=0.9, max_features=1000).fit_transform(df_all['taglist'])
    del df_all['taglist'], df_all['taglist_x'], df_all['label']

    # 当采用不允许缺失的模型时，以平均值填充缺失值
    if fill is True:
        df_all.fillna(df_all.mean(), inplace=True)
    df_final = sparse.hstack((df_all.values, tag_cv), format='csr', dtype='float32')
    print('数据预处理已完成')
    return train_y, test_due_amt, sub, df_final


def feature_selcetion(df_final,train_y,num=train_num):
    trn_x, trn_y = df_final[:num], train_y['label'][:num]
    x_train, x_val, y_train, y_val = train_test_split(trn_x, trn_y, train_size=0.8, random_state=2019, stratify=trn_y)
    clf = LGBMClassifier(learning_rate=0.05, n_estimators=10000, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                         random_state=2019)
    t = time.time()
    clf.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_val, y_val)],early_stopping_rounds=20, verbose=5)
    print('runtime: {}\n'.format(time.time() - t))
    feature_impo = pd.DataFrame(sorted(zip(clf.feature_importances_, range(df_final.shape[1]))),columns=['Value', 'Feature'])
    feature_impo.to_csv(path + 'feature_impo.csv', index=False)
    df_final_selected=pd.read_csv(path + 'feature_impo.csv')
    fea_num_0=df_final.shape[1]
    fea_num_1=fea_num_0-1000
    head_1000_fea=feature_impo[fea_num_1:fea_num_0]['Feature'].values.tolist()
    df_final = df_final[:, head_1000_fea]
    print('特征选择已完成')
    return df_final


def custom_loss(y_true,y_pre):
    loss_amt=0
    shape=y_true.shape[0]
    if shape>train_num/6:
        for i in range(shape):
            prob=(1-y_pre[y_true[i]*shape+i])
            amt=train_amt[i]
            loss_amt+=np.square(prob*amt)/shape
    else:
        for i in range(shape):
            prob=(1-y_pre[y_true[i]*shape+i])
            amt=val_amt[i]
            loss_amt+=np.square(prob*amt)/shape
    return 'loss_amt',np.sqrt(loss_amt),False


def single_model(df_final, train_y,weight=None,metric=None):
    train_values, test_values = df_final[:train_num], df_final[test_num:]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
    clf = LGBMClassifier(learning_rate=0.05, n_estimators=10000, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, random_state=2019)
    test_pred_prob = np.zeros((test_values.shape[0], 33))
    for i, (trn_idx, val_idx) in enumerate(skf.split(train_values, train_y['label'])):
        print(i, 'fold...')
        t = time.time()
        trn_x, trn_y = train_values[trn_idx], train_y['label'][trn_idx]
        val_x, val_y = train_values[val_idx], train_y['label'][val_idx]
        train_amt, val_amt = train_y['due_amt'][trn_idx].values, train_y['due_amt'][val_idx].values
        clf.fit(trn_x, trn_y, eval_set=[(trn_x, trn_y), (val_x, val_y)],sample_weight=weight,
                eval_metric=metric, early_stopping_rounds=100, verbose=5)
        test_pred_prob += clf.predict_proba(test_values, num_iteration=clf.best_iteration_) / skf.n_splits
        print('runtime: {}\n'.format(time.time() - t))
    print('单模型拟合已完成')
    return test_pred_prob


def get_stacking(clf, x_train, y_train, x_test, n_folds=5):
    print(clf)
    second_level_train_set = pd.DataFrame()
    second_level_test_temp = pd.DataFrame(index=range(x_test.shape[0]))
    second_level_test_set = pd.DataFrame()
    kf = KFold(n_splits=n_folds)
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        print(i)
        x_tra, y_tra = x_train[train_index], y_train[train_index]
        x_tst, y_tst = x_train[test_index], y_train[test_index]
        clf.fit(x_tra, y_tra)
        split_train_set = pd.DataFrame(clf.predict_proba(x_tst), index=test_index)
        second_level_train_set = pd.concat([second_level_train_set, split_train_set], axis=0)
        split_test_set = pd.DataFrame(clf.predict_proba(x_test))
        col_list = range(33 * i, 33 * i + 33)
        split_test_set.columns = col_list
        second_level_test_temp = pd.concat([second_level_test_temp, split_test_set], axis=1)

    for i in range(0, 33):
        second_level_test_set[str(i)] = second_level_test_temp[[i, i + 33]].mean(axis=1)
    return second_level_train_set, second_level_test_set


def stacking(df_final, train_y):
    trn_x, tran_y, test_x = df_final[:train_num], train_y[:train_num]['label'], df_final[test_num:]
    xgb_model = XGBClassifier(random_state=2019)
    lgb_model = LGBMClassifier(random_state=2019)
    rf_model = RandomForestClassifier(random_state=2019)
    svc_model = SVC(probability=True)
    train_sets = pd.DataFrame()
    test_sets = pd.DataFrame()
    for clf in [lgb_model, rf_model]:
        train_set, test_set = get_stacking(clf, trn_x, tran_y, test_x)
        train_sets = pd.concat([train_sets, train_set], axis=1)
        test_sets = pd.concat([test_sets, test_set], axis=1)
    xgb_model.fit(train_sets, train_y)
    test_pred_prob = xgb_model.predict_proba(test_sets)
    print('stacking模型已完成')
    return test_pred_prob


def out_put(test_pred_prob, sub, ratio_method=False,name=None):
    prob_cols = ['prob_{}'.format(i) for i in range(33)]
    for i, f in enumerate(prob_cols):
        sub[f] = test_pred_prob[:, i]
    sub_example = pd.read_csv(path+'submission.csv', parse_dates=['repay_date'])
    sub_example = sub.merge(sub_example, on='listing_id', how='left')
    sub_example['days'] = (sub_example['due_date'] - sub_example['repay_date']).dt.days
    sub_example['days'] = sub_example['days'].astype(int, inplace=True)
    sub_example.reset_index(inplace=True)
    #加入将明显逾期的标的抹去规则
    if ratio_method:
        sub_example['ratio'] = 1 / (1 - sub_example['prob_32'])
        prob_cols_repay = ['prob_{}'.format(i) for i in range(32)]
        measure_1 = (sub_example['prob_32'].mean()) * 2.5
        measure_0 = (sub_example['prob_32'].mean()) * 0.4
        for f in prob_cols_repay:
            sub_example[f][sub_example.prob_32 > measure_1] = 0
            sub_example[f][sub_example.prob_32 < measure_0] = sub_example[f] * sub_example['ratio']
    test_prob_value = sub_example[prob_cols].values
    test_labels_value = sub_example['days'].values
    test_prob = [test_prob_value[i][test_labels_value[i]] for i in range(test_prob_value.shape[0])]
    sub_example['repay_amt'] = sub_example['due_amt'] * test_prob
    sub_example[['listing_id', 'repay_date', 'repay_amt']].to_csv(path + name + '_sub.csv', index=False)
    print('结果导出已完成')


if __name__ == "__main__":
    train_y, test_due_amt, sub, df_final = pre_process(fill=True)
    #df_final_selected=feature_selcetion(df_final,num=train_num)
    #test_pred_prob = single_model(df_final, train_y, weight=None,metric=None)
    test_pred_prob = stacking(df_final,train_y)
    out_put(test_pred_prob,sub,name='xxxxxx')

