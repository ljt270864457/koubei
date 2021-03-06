#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/2 7:29 PM
# @Author  : liujiatian
# @File    : stat_day.py


import numpy as np
import pandas as pd


def stat_day_model():
    '''
    # TODO 需要构造天气数据
    统计模型 => stat_day.csv
    shop_id date 浏览量 购买量 营业时间 周 是否是节假日 weekid
    :return:
    '''
    # 处理用户支付数据
    df_pay = pd.read_csv('../data/user_pay.txt', header=None)
    df_pay.columns = ['user_id', 'shop_id', 'timestamp']
    df_pay['date'] = df_pay['timestamp'].apply(lambda x: x[:10])
    df_pay = df_pay[(df_pay['date'] >= '2016-09-06') &
                    (df_pay['date'] <= '2016-10-31')]
    df_pay = df_pay[(df_pay['date'] > '2016-10-07') |
                    (df_pay['date'] < '2016-10-01')]
    df_agg_pay = df_pay.groupby(['shop_id', 'date']).agg({
        'timestamp': ['min', 'max', 'size'],
    }).reset_index()
    df_agg_pay.columns = ['shop_id', 'date', 'min_ts', 'max_ts', 'pay_count']
    df_agg_pay['operate_hour'] = ((pd.to_datetime(
        df_agg_pay['max_ts']) - pd.to_datetime(df_agg_pay['min_ts'])) / np.timedelta64(1, 'h')).astype(int)
    df_agg_pay = df_agg_pay.drop(['min_ts', 'max_ts'], axis=1)
    # 处理浏览数据
    df_browser = pd.read_csv('../data/user_view.txt')
    df_browser.columns = ['user_id', 'shop_id', 'timestamp']

    df_browser['date'] = df_browser['timestamp'].apply(lambda x: x[:10])
    df_browser = df_browser[(df_browser['date'] >= '2016-09-06') & (df_browser['date'] <= '2016-10-31')]
    df_browser = df_browser[(df_browser['date'] > '2016-10-07') | (df_browser['date'] < '2016-10-01')]
    browser_stat_day = df_browser.groupby(['shop_id', 'date']).size().reset_index()
    browser_stat_day.columns = ['shop_id', 'date', 'browser_count']
    dim_list = []
    # 构建日期维度表
    shop_id = df_agg_pay['shop_id'].unique()
    date_list = df_agg_pay['date'].unique()
    for _id in shop_id:
        for date in date_list:
            data = {'shop_id': _id, 'date': date}
            dim_list.append(data)
    df_dim = pd.DataFrame(dim_list)
    df = pd.merge(df_dim, df_agg_pay, how='left').fillna(0)
    df = pd.merge(df, browser_stat_day, how='left').fillna(0)
    df['dayofweek'] = pd.to_datetime(df['date']).dt.dayofweek

    # 需要处理节假日及调休
    def deal_holiday(x):
        result = 1 if pd.to_datetime(x).strftime("%w") in ['0', '6'] else 0
        if x in ['2016-09-15', '2016-09-16', '2016-09-17']:
            result = 1
        if x in ['2016-09-18', '2016-10-08', '2016-10-09']:
            result = 0
        return result

    df['is_holiday'] = df['date'].apply(deal_holiday)
    day_week_config = {
        '2016-09-06': 1,
        '2016-09-07': 1,
        '2016-09-08': 1,
        '2016-09-09': 1,
        '2016-09-10': 1,
        '2016-09-11': 1,
        '2016-09-12': 1,

        '2016-09-13': 2,
        '2016-09-14': 2,
        '2016-09-15': 2,
        '2016-09-16': 2,
        '2016-09-17': 2,
        '2016-09-18': 2,
        '2016-09-19': 2,

        '2016-09-20': 3,
        '2016-09-21': 3,
        '2016-09-22': 3,
        '2016-09-23': 3,
        '2016-09-24': 3,
        '2016-09-25': 3,
        '2016-09-26': 3,

        '2016-09-27': 4,
        '2016-09-28': 4,
        '2016-09-29': 4,
        '2016-09-30': 4,
        '2016-10-08': 4,
        '2016-10-09': 4,
        '2016-10-10': 4,

        '2016-10-11': 5,
        '2016-10-12': 5,
        '2016-10-13': 5,
        '2016-10-14': 5,
        '2016-10-15': 5,
        '2016-10-16': 5,
        '2016-10-17': 5,

        '2016-10-18': 6,
        '2016-10-19': 6,
        '2016-10-20': 6,
        '2016-10-21': 6,
        '2016-10-22': 6,
        '2016-10-23': 6,
        '2016-10-24': 6,

        '2016-10-25': 7,
        '2016-10-26': 7,
        '2016-10-27': 7,
        '2016-10-28': 7,
        '2016-10-29': 7,
        '2016-10-30': 7,
        '2016-10-31': 7
    }
    df['week_id'] = df['date'].apply(lambda x: day_week_config.get(x))
    for column in ['pay_count', 'operate_hour', 'browser_count', 'dayofweek', 'is_holiday']:
        df[column] = df[column].astype(int)
    df.to_csv('../data/stat_day.csv', index=False)


if __name__ == '__main__':
    stat_day_model()
