# coding=utf-8
import os
import re
from datetime import datetime

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


def smooth_sequence(sequence, a=0.5):
    '''
    一次指数平滑
    a越小，平滑性越好
    '''
    result = []
    sequence = list(sequence)
    for i in range(len(sequence)):
        y_true = sequence[i]
        if i == 0:
            y_hat = round((sequence[0] + sequence[1] + sequence[2]) / 3)
        y_hat = round((1 - a) * y_hat + a * y_true)
        result.append(y_hat)
    return result


def judge_zero(x):
    result = 0
    for _ in range(1, 22):
        if x[_] == 0:
            result = 1
            break
    return result


def get_holiday():
    result = dict()
    df = pd.read_csv(f'{DATA_DIR}/stat_day.csv', usecols=['date', 'is_holiday'])
    df = df.drop_duplicates()
    tmp_list = df.to_dict(orient='records')
    for each in tmp_list:
        _date = each.get('date')
        is_holiday = each.get('is_holiday')
        result[_date] = is_holiday
    return result


def construct_data(df_pay, train_start_date, train_end_date, test_start_date, test_end_date, holiday_mappings):
    '''
    构造数据集
    '''
    df = df_pay[(df_pay['date'] >= train_start_date) & (df_pay['date'] <= test_end_date)]
    df = pd.pivot_table(df, index='shop_id', columns='date', values='pay_count').reset_index()

    # 对销量数据删除存在0的样本，并且对21天的销量进行一次指数平滑，平滑参数为0.5

    df['is_delete'] = df.apply(judge_zero, axis=1)
    df = df[df['is_delete'] == 0]
    df.drop('is_delete', axis=1, inplace=True)

    df_sample = df.iloc[:, :22]
    df_label = df.iloc[:, 22:]

    columns = df_sample.columns
    result = []
    for row_id, row in df_sample.iterrows():
        sequence = list(row[1:22])
        new_sequence = smooth_sequence(sequence, 0.5)
        data = [row[0]]
        data.extend(new_sequence)
        data.extend(row[22:-1])
        result.append(data)
    df_sample = pd.DataFrame(result)
    df_sample.columns = columns

    # 构建商铺进本特征
    df_shop_feature = gen_shop_feature(train_start_date, train_end_date)
    df_sample = pd.merge(df_sample, df_shop_feature, on=['shop_id'], how='left')

    # 构建节假日特征
    sample_columns = [_ for _ in df_sample.columns if re.search('\d{4}-\d{2}-\d{2}', _)]
    label_columns = [_ for _ in df_label.columns if re.search('\d{4}-\d{2}-\d{2}', _)]
    date_columns = sorted(sample_columns + label_columns)
    for index, each in enumerate(date_columns):
        df_sample[f'day_{index + 1}_is_holiday'] = holiday_mappings.get(each)
        df_sample[f'day_{index + 1}_dayofweek'] = datetime.strptime(each, '%Y-%m-%d').weekday()
    return df_sample, df_label


def run_construct():
    df_pay = pd.read_csv(f'{DATA_DIR}/stat_day.csv')
    holidays = get_holiday()
    for k, v in DATA_CONFIG.items():
        train_start_date = v.get('train_start_date')
        train_end_date = v.get('train_end_date')
        test_start_date = v.get('test_start_date')
        test_end_date = v.get('test_end_date')
        print(train_start_date, train_end_date, test_start_date, test_end_date)

        df_sample, df_label = construct_data(df_pay, train_start_date, train_end_date, test_start_date, test_end_date,
                                             holidays)

        df_sample.to_csv(f'{DATA_DIR}/sample_{k}_{train_start_date}_{test_end_date}.csv', index=False)
        df_label.to_csv(f'{DATA_DIR}/label_{k}_{train_start_date}_{test_end_date}.csv', index=False)


if __name__ == '__main__':
    run_construct()
