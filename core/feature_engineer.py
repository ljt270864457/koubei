#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/3 4:41 PM
# @Author  : liujiatian
# @File    : feature_engineer.py

import pandas as pd


def gen_shop_feature(start_date, end_date):
    '''
    1. 商店基本信息：商家类别，消费等级，评论数，浏览量，工作日平均销量，评分
    2. 统计指标 节假日每天营业时长，工作日营业时长，节假日销量情况，工作日销量情况，节假日浏览量，工作日浏览量，节假日/工作日
    :return:
    '''
    df = pd.read_csv('../data/stat_day.csv')
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    # 营业时长统计
    df_operate = df[df['operate_hour'] > 0].groupby(['shop_id', 'is_holiday'])['operate_hour'].agg(
        ['min', 'max', 'mean', 'sum']).reset_index()
    df_operate = pd.pivot_table(df_operate, index='shop_id', columns='is_holiday',
                                values=['min', 'max', 'mean', 'sum']).reset_index()
    df_operate.columns = ['shop_id', 'operate_weekday_max', 'operate_holiday_max', 'operate_weekday_mean',
                          'operate_holiday_mean', 'operate_weekday_min', 'operate_holiday_min', 'operate_weekday_sum',
                          'operate_holiday_sum']
    df_operate['operate_mean_holiday_div_weekday'] = df_operate['operate_holiday_mean'] / df_operate[
        'operate_weekday_mean']
    df_operate['operate_total_holiday_div_weekday'] = df_operate['operate_holiday_sum'] / df_operate[
        'operate_weekday_sum']

    # 销量统计
    df_pay = df[df['pay_count'] > 0].groupby(['shop_id', 'is_holiday'])['pay_count'].agg(
        ['min', 'max', 'mean', 'std']).reset_index()
    df_pay = pd.pivot_table(df_pay, index='shop_id', columns='is_holiday',
                            values=['min', 'max', 'mean', 'std']).reset_index()
    df_pay.columns = ['shop_id', 'pay_weekday_max', 'pay_holiday_max', 'pay_weekday_mean', 'pay_holiday_mean',
                      'pay_weekday_min', 'pay_holiday_min', 'pay_weekday_sum', 'pay_holiday_sum']
    df_pay['pay_mean_holiday_div_weekday'] = df_pay['pay_holiday_mean'] / df_pay['pay_weekday_mean']
    df_pay['pay_total_holiday_div_weekday'] = df_pay['pay_holiday_sum'] / df_pay['pay_weekday_sum']

    # 浏览量统计
    df_browser = df[df['browser_count'] > 0].groupby(['shop_id', 'is_holiday'])['browser_count'].agg(
        ['min', 'max', 'mean', 'sum']).reset_index()
    df_browser = pd.pivot_table(df_browser, index='shop_id', columns='is_holiday',
                                values=['min', 'max', 'mean', 'sum']).reset_index()
    df_browser.columns = ['shop_id', 'browser_weekday_max', 'browser_holiday_max', 'browser_weekday_mean',
                          'browser_holiday_mean', 'browser_weekday_min', 'browser_holiday_min', 'browser_weekday_sum',
                          'browser_holiday_sum']
    df_browser['pay_mean_holiday_div_weekday'] = df_browser['browser_holiday_mean'] / df_browser['browser_weekday_mean']
    df_browser['pay_total_holiday_div_weekday'] = df_browser['browser_holiday_sum'] / df_browser['browser_weekday_sum']

    # 商铺基本信息
    df_shop = pd.read_csv('../data/shop_info.txt', header=None)
    df_shop.columns = ['shop_id', 'city_name', 'location_id', 'per_pay',
                       'score', 'comment_cnt', 'shop_level', 'cate_1_name', 'cate_2_name', 'cate_3_name']
    df_city_dummies = pd.get_dummies(df_shop['city_name'])
    df_shop = df_shop.join(df_city_dummies)

    cate1_dummies = pd.get_dummies(df_shop['cate_1_name'], prefix='cate1')
    cate2_dummies = pd.get_dummies(df_shop['cate_2_name'], prefix='cate2')
    cate3_dummies = pd.get_dummies(df_shop['cate_3_name'], prefix='cate3')

    df_shop = df_shop.join(cate1_dummies)
    df_shop = df_shop.join(cate2_dummies)
    df_shop = df_shop.join(cate3_dummies)

    df_shop = df_shop.drop(['city_name', 'location_id', 'cate_1_name', 'cate_2_name', 'cate_3_name'], axis=1)

    # 合并特征
    df_final = pd.merge(df_shop, df_operate, on='shop_id', how='left')
    df_final = pd.merge(df_final, df_pay, on='shop_id', how='left')
    df_final = pd.merge(df_final, df_browser, on='shop_id', how='left')
    df_final['holiday_pay_total_div_browser'] = df_final['pay_holiday_sum'] / df_final['browser_holiday_sum']
    df_final['weekday_pay_total_div_browser'] = df_final['pay_weekday_sum'] / df_final['browser_weekday_sum']
    df_final['holiday_pay_mean_div_browser'] = df_final['pay_holiday_mean'] / df_final['browser_holiday_mean']
    df_final['weekday_pay_mean_div_browser'] = df_final['pay_weekday_mean'] / df_final['browser_weekday_mean']
    # df_final.to_csv('../data/feature_shop.csv', index=False)
    return df_final
