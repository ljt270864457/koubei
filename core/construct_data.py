import os
import pandas as pd

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
        'train_start_date': '2016-09-10',
        'train_end_date': '2016-09-30',
        'test_start_date': '2016-10-08',
        'test_end_date': '2016-10-14'
    },
    'train3': {
        'train_start_date': '2016-09-03',
        'train_end_date': '2016-09-23',
        'test_start_date': '2016-09-24',
        'test_end_date': '2016-09-30'
    },
}


def construct_data(df_pay, train_start, train_end, test_start, test_end):
    '''
    构造数据集
    '''
    # 全量时间窗口
    all_date_range = [x.strftime('%Y-%m-%d') for x in pd.date_range(train_start, test_end)]
    # 忽略十一假期
    filter_date_range = [x.strftime('%Y-%m-%d') for x in pd.date_range('2016-10-01', '2016-10-07')]
    all_date_range = list(filter(lambda x: x not in filter_date_range, all_date_range))

    # 训练集时间窗口
    train_date_range = all_date_range[:21]
    # 测试集时间窗口
    test_date_range = all_date_range[21:]

    # 构建店铺-日期维度
    df_date = pd.DataFrame(all_date_range, columns=['date'])
    df_date['key'] = 0
    assert df_date.shape[0] == 28
    df_shop = pd.DataFrame(range(1, 2001), columns=['shop_id'])
    df_shop['key'] = 0
    df_dim = pd.merge(df_shop, df_date, on='key')
    df_dim = df_dim[['shop_id', 'date']]
    assert df_dim.shape[0] == 2000 * 28

    # 日期维度与实际支付数据进行合并
    df = pd.merge(df_dim, df_pay, how='left').fillna(0)
    df_train = pd.pivot_table(df[df['date'].isin(train_date_range)], index='shop_id', columns='date',
                              values='pay_count').reset_index()
    #     columns = ['shop_id']
    #     columns.extend([f'day_{_}' for _ in range(1,22)])
    #     df_train.columns = columns
    #     df_train.head()
    df_label = df[df['date'].isin(test_date_range)].groupby('shop_id')['pay_count'].sum().reset_index()
    df_final = pd.merge(df_train, df_label, on='shop_id')
    return df_final


def run_construct():
    df_pay = pd.read_csv(f'{DATA_DIR}/pay_stat_day.csv')
    for k, v in DATA_CONFIG.items():
        train_start_date = v.get('train_start_date')
        train_end_date = v.get('train_end_date')
        test_start_date = v.get('test_start_date')
        test_end_date = v.get('test_end_date')
        tmp = construct_data(df_pay, train_start_date,
                             train_end_date, test_start_date, test_end_date)
        tmp.to_csv(f'{DATA_DIR}/{k}_{train_start_date}_{test_end_date}.csv', index=False, header=None)


if __name__ == '__main__':
    run_construct()
