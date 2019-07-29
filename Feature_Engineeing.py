import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import datetime

short_path=''

#读取train信息，并生成label
df_train=pd.read_csv(short_path+'train.csv',parse_dates=['auditing_date', 'due_date', 'repay_date'],encoding='utf-8')
df_train['repay_amt'] = df_train['repay_amt'].apply(lambda x: x if x != '\\N' else 0).astype('float')
df_train['label'] = df_train[['due_date', 'repay_date']].apply(lambda x: (x['due_date']-x['repay_date']).days if x['repay_date'] != '\\N' else 32, axis=1)

#读取test文本并与train拼接
df_test=pd.read_csv(short_path+'test.csv',parse_dates=['auditing_date', 'due_date'],encoding='utf-8')
df_all=pd.concat([df_train,df_test],axis=0,ignore_index=True)
df_all['day_term']=(df_all['due_date']-df_all['auditing_date']).dt.days

#录入list_info信息
df_listing_info = pd.read_csv(short_path+'listing_info.csv',parse_dates=['auditing_date'],encoding='utf-8')
df_listing_info.sort_values(by=['user_id','auditing_date'],inplace=True,ascending=False)

#读取repay_log信息
df_repay_log=pd.read_csv(short_path+'user_repay_logs.csv',parse_dates=['due_date','repay_date'])
df_repay_log = df_repay_log[df_repay_log['order_id'] == 1].reset_index(drop=True)
df_repay_log['repay'] = df_repay_log['repay_date'].astype('str').apply(lambda x: 1 if x != '2200-01-01' else 0)
df_repay_log['advance']=df_repay_log[['due_date','repay_date']].apply(lambda x:(x['due_date'] - x['repay_date']).days if (x['due_date'] - x['repay_date']).days>=0 else -1,axis=1)

#读取user_behavior信息
df_user_behavior=pd.read_csv(short_path+'user_behavior_logs.csv',parse_dates=['behavior_time'],index_col=0)
df_user_behavior['Hour']=df_user_behavior['behavior_time'].dt.hour

#基于上述信息构造部分新特征
info_list=['shape','mean','std','recent','last','advance_mean','advance_std','behavior_count','behavior_1','behavior_2','behavior_3','behavior_dawn']
def add_info(user_id,auditing_date):
    #基于成交日期选择新特征组合
    group_0=df_listing_info.loc[(df_listing_info['user_id']==user_id) & (df_listing_info['auditing_date']<auditing_date)].reset_index(drop=True)
    group_1=(df_repay_log.loc[(df_repay_log['user_id']==user_id) & (df_repay_log['repay_date']<auditing_date),'advance'])
    group_2=(df_user_behavior.loc[(df_user_behavior['user_id']==user_id) & (df_user_behavior['behavior_time']<auditing_date)])
    sixty_day_time=auditing_date-datetime.timedelta(days=60)
    if group_0.empty==False:
        last_t=group_0.loc[0,'auditing_date']
    else:
        last_t=np.nan
    #构造list_info新特征
    shape=group_0.shape[0]
    mean=group_0['principal'].mean()
    std=group_0['principal'].std()
    recent=group_0.loc[group_0['auditing_date']>sixty_day_time].shape[0]
    #构造repay_info新特征
    advance_mean=group_1.mean()
    advance_std=group_1.std()
    #构造user_behavior新特征
    behavior_count=group_2.shape[0]
    behavior_1=sum(group_2['behavior_type'].apply(lambda x: 1 if x==1 else 0))
    behavior_2=sum(group_2['behavior_type'].apply(lambda x: 1 if x==2 else 0))
    behavior_3=sum(group_2['behavior_type'].apply(lambda x: 1 if x==3 else 0))
    behavior_dawn=sum(group_2['Hour'].apply(lambda x:1 if 0<=x<6 else 0))
    return [shape,mean,std,recent,last_t,advance_mean,advance_std,
            behavior_count,behavior_1,behavior_2,behavior_3,behavior_dawn]

add_info=(df_all.apply(lambda row:add_info(row['user_id'],row['auditing_date']),axis=1))
df_add_fea=pd.DataFrame()
for i,f in enumerate(info_list):
    df_add_fea[f] = add_info.apply(lambda x: x[i])
df_all=pd.concat([df_all,df_add_fea],axis=1)
del df_add_fea

del df_listing_info['user_id'], df_listing_info['auditing_date']
df_all = df_all.merge(df_listing_info, on='listing_id', how='left')

df_all['last'] = pd.to_datetime(df_all['last'],format='%Y-%m-%d %H:%M:%S')

df_all['time_difference']=(df_all['auditing_date']-df_all['last']).dt.days
df_all['delta_principal']=(df_all['principal']-df_all['mean'])/df_all['mean']

df_user_info=pd.read_csv(short_path+'user_info.csv',parse_dates=['reg_mon', 'insertdate'])
df_user_info.rename(columns={'insertdate': 'info_insert_date'}, inplace=True)
df_user_info = df_user_info.sort_values(by='info_insert_date', ascending=True).drop_duplicates('user_id').reset_index(drop=True)
df_all = df_all.merge(df_user_info, on='user_id', how='left')

df_user_tag = pd.read_csv(short_path+'user_taglist.csv', parse_dates=['insertdate'])
df_user_tag.rename(columns={'insertdate': 'tag_insert_date'}, inplace=True)
df_user_tag['auditing_date']=df_user_tag['tag_insert_date']+datetime.timedelta(days=1)
df_user_tag = df_user_tag.reset_index(drop=True)
df_all = df_all.merge(df_user_tag, on=['user_id','auditing_date'], how='left')

df_user_tag_2 = df_user_tag.sort_values(by='tag_insert_date', ascending=False).drop_duplicates('user_id').reset_index(drop=True)
df_all = df_all.merge(df_user_tag_2, on='user_id', how='left')
df_all.loc[((df_all['taglist_x'].isnull().values==True) & (df_all['tag_insert_date_y']<df_all['auditing_date_x'])),'taglist_x']=df_all.loc[((df_all['taglist_x'].isnull().values==True) & (df_all['tag_insert_date_y']<df_all['auditing_date_x'])),'taglist_y']
df_all['new_tag'] = df_all[['tag_insert_date_x', 'auditing_date_x']].apply(lambda x: 1 if (x['auditing_date_x']-x['tag_insert_date_x']).days==1 else 0, axis=1)
del df_all['taglist_y'], df_all['tag_insert_date_y'], df_all['tag_insert_date_x'], df_all['auditing_date_y']

df_all['same_province_label'] = df_all.apply(lambda x: 0 if x['id_province'] == x['cell_province'] else 1, axis=1)

df_all['due_month']=df_all['due_date'].dt.month
df_all['due_day']=df_all['due_date'].dt.day
df_all['due_dayofweek']=df_all['due_date'].dt.dayofweek
df_all['aud_dayofweek']=df_all['auditing_date_x'].dt.dayofweek

df_all['auditing_mon']=df_all['auditing_date_x'].dt.month
df_all['auditing_year']=df_all['auditing_date_x'].dt.year
mon_term_mean=df_all.groupby(['auditing_year','auditing_mon','term'])['rate'].median()
df_all=df_all.merge(mon_term_mean, on=['auditing_year','auditing_mon','term'], how='left')
df_all['rate_delta']=(df_all['rate_y']-df_all['rate_x'])/df_all['rate_y']

from scipy.stats import kurtosis
df_repay_log['repay'] = df_repay_log['repay_date'].astype('str').apply(lambda x: 1 if x != '2200-01-01' else 0)
df_repay_log['early_repay_days'] = (df_repay_log['due_date'] - df_repay_log['repay_date']).dt.days
df_repay_log['early_repay_days'] = df_repay_log['early_repay_days'].apply(lambda x: x if x >= 0 else -1)
for f in ['listing_id', 'order_id', 'due_date', 'repay_date', 'repay_amt']:
    del df_repay_log[f]
group = df_repay_log.groupby('user_id', as_index=False)
df_repay_log = df_repay_log.merge(group['repay'].agg({'repay_mean': 'mean'}), on='user_id', how='left')
df_repay_log = df_repay_log.merge(group['early_repay_days'].agg({
        'early_repay_days_max': 'max', 'early_repay_days_median': 'median', 'early_repay_days_sum': 'sum',
        'early_repay_days_mean': 'mean', 'early_repay_days_std': 'std'
    }), on='user_id', how='left')
df_repay_log = df_repay_log.merge(group['due_amt'].agg({'due_amt_max': 'max', 'due_amt_min': 'min', 'due_amt_median': 'median',
                                                        'due_amt_mean': 'mean', 'due_amt_sum': 'sum', 'due_amt_std': 'std',
                                                        'due_amt_skew': 'skew', 'due_amt_kurt': kurtosis, 'due_amt_ptp': np.ptp
                                                       }), on='user_id', how='left')
del df_repay_log['repay'], df_repay_log['early_repay_days'], df_repay_log['due_amt']
df_repay_log = df_repay_log.drop_duplicates('user_id').reset_index(drop=True)

df_all = df_all.merge(df_repay_log, on='user_id', how='left')

df_all.to_csv(short_path+'features.csv',encoding='gbk')