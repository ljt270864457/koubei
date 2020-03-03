import os
import pandas as pd
from core.feature_engineer import gen_shop_feature

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
DATA_CONFIG = {
    'test': {
        'train_start_date': '2016-09-27',
        'train_end_date': '2016-10-24',
        'test_start_date': '2016-10-25',
        'test_end_date': '2016-10-31'
    },
    'train1': {
        'train_start_date': '2016-09-20',
        'train_end_date': '2016-10-17',
        'test_start_date': '2016-10-18',
        'test_end_date': '2016-10-24'
    },
    'train2': {
        'train_start_date': '2016-09-13',
        'train_end_date': '2016-10-10',
        'test_start_date': '2016-10-11',
        'test_end_date': '2016-10-17'
    },
    'train3': {
        'train_start_date': '2016-09-06',
        'train_end_date': '2016-09-26',
        'test_start_date': '2016-09-27',
        'test_end_date': '2016-10-10'
    },
}


def construct_data(df_pay, train_start_date, train_end_date, test_start_date, test_end_date):
    '''
    构造数据集
    '''
    df = df_pay[(df_pay['date'] >= train_start_date) & (df_pay['date'] <= test_end_date)]
    df = pd.pivot_table(df, index='shop_id', columns='date', values='pay_count').reset_index()
    df_sample = df.iloc[:, :22]
    df_label = df.iloc[:, 22:]

    df_shop_feature = gen_shop_feature(train_start_date, train_end_date)
    df_sample = pd.merge(df_sample, df_shop_feature, on=['shop_id'], how='left')
    return df_sample, df_label


def run_construct():
    df_pay = pd.read_csv(f'{DATA_DIR}/stat_day.csv')
    for k, v in DATA_CONFIG.items():
        train_start_date = v.get('train_start_date')
        train_end_date = v.get('train_end_date')
        test_start_date = v.get('test_start_date')
        test_end_date = v.get('test_end_date')
        df_sample, df_label = construct_data(df_pay, train_start_date, train_end_date, test_start_date, test_end_date)
        df_sample.to_csv(f'{DATA_DIR}/sample_{k}_{train_start_date}_{test_end_date}.csv', index=False)
        df_label.to_csv(f'{DATA_DIR}/label_{k}_{train_start_date}_{test_end_date}.csv', index=False)


if __name__ == '__main__':
    run_construct()
