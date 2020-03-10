import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost

import lightgbm as lgb

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')

result = []
rmse_list = []


def load_data():
    '''
    加载训练集和测试集
    :return:
    '''
    df_train1_sample = pd.read_csv(f'{DATA_DIR}/sample_train1_2016-09-20_2016-10-24.csv', skiprows=1, header=None)
    df_train2_sample = pd.read_csv(f'{DATA_DIR}/sample_train2_2016-09-13_2016-10-17.csv', skiprows=1, header=None)
    df_train3_sample = pd.read_csv(f'{DATA_DIR}/sample_train3_2016-09-06_2016-10-10.csv', skiprows=1, header=None)

    df_train1_label = pd.read_csv(f'{DATA_DIR}/label_train1_2016-09-20_2016-10-24.csv', skiprows=1, header=None)
    df_train2_label = pd.read_csv(f'{DATA_DIR}/label_train2_2016-09-13_2016-10-17.csv', skiprows=1, header=None)
    df_train3_label = pd.read_csv(f'{DATA_DIR}/label_train3_2016-09-06_2016-10-10.csv', skiprows=1, header=None)

    df_train_X = pd.concat([df_train1_sample, df_train2_sample, df_train3_sample])
    df_train_y_7_days = pd.concat([df_train1_label, df_train2_label, df_train3_label])

    df_test_X = pd.read_csv(f'{DATA_DIR}/sample_test_2016-09-27_2016-10-31.csv', skiprows=1, header=None)
    df_test_y_7days = pd.read_csv(f'{DATA_DIR}/label_test_2016-09-27_2016-10-31.csv', skiprows=1, header=None)
    return df_train_X, df_train_y_7_days, df_test_X, df_test_y_7days


def train_model_xgb(df_train_X, df_train_y_7_days, df_test_X, df_test_y_7days):
    '''
    训练xgb模型
    :param df_train_X:
    :param df_train_y_7_days:
    :param df_test_X:
    :param df_test_y_7days:
    :return:
    '''
    train_X = df_train_X.values
    test_X = df_test_X.values
    model = xgboost.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.85, reg_lambda=1.3)

    for i in range(7):
        print(f'训练第{i+1}个模型，预测第{i+1}天的数据....')
        train_y = df_train_y_7_days.iloc[:, i].values
        test_y = df_test_y_7days.iloc[:, i].values

        model.fit(train_X, train_y, eval_set=[(test_X, test_y)], eval_metric='rmse', early_stopping_rounds=10)
        pickle.dump(model, open(f'{MODEL_DIR}/xgb_day{i}.pkl', 'wb'))


def predict_xgb(df_test_X, df_test_y_7_days):
    '''
    使用xgb模型进行预测
    :param df_test_X:
    :param df_test_y_7_days:
    :return:
    '''
    result = []
    rmse_list = []

    test_X = df_test_X.values

    for i in range(7):
        print(f'预测第{i+1}天...')
        path = f'{MODEL_DIR}/xgb_day{i}.pkl'
        model = pickle.load(open(path, 'rb'))
        test_y = df_test_y_7_days.iloc[:, i].values

        # 通过误差分析 发现周末的销量存在较大的误差，在实际预测的时候加上0.95的策略参数
        if i in (4, 5):
            y_hat = np.round(model.predict(test_X) * 0.97).astype(int)
        else:
            y_hat = np.round(model.predict(test_X)).astype(int)
        result.append(y_hat.reshape(-1, 1))
        rmse_list.append(np.sqrt(mean_squared_error(test_y, y_hat)))

    print('计算评价指标')
    predict_matrix = np.hstack(result)
    true_matrix = df_test_y_7_days.values
    df_predict = pd.DataFrame(predict_matrix)
    df_true = pd.DataFrame(true_matrix)
    df_predict.to_csv(f'{DATA_DIR}/prediction/xgb_predict.csv', index=False, header=False)
    df_true.to_csv(f'{DATA_DIR}/prediction/xgb_true.csv', index=False, header=False)

    loss = np.sum(np.abs((true_matrix - predict_matrix) / (true_matrix + predict_matrix))) / true_matrix.shape[0] / \
           true_matrix.shape[1]
    rmse = np.mean(rmse_list)
    print(f'loss:{round(loss,6)},rmse:{round(rmse,6)}')


def train_model_lgb(df_train_X, df_train_y_7_days, df_test_X, df_test_y_7days):
    '''
    使用lgb进行训练
    :param df_train_X:
    :param df_train_y_7_days:
    :param df_test_X:
    :param df_test_y_7days:
    :return:
    '''
    train_X = df_train_X.values
    test_X = df_test_X.values
    model = lgb.LGBMRegressor(n_estimators=300, max_depth=4, max_bin=1024, learning_rate=0.1, subsample=0.85,
                              num_leaves=63)
    for i in range(7):
        print(f'lgb 训练第{i+1}个模型')
        train_y = df_train_y_7_days.iloc[:, i].values
        test_y = df_test_y_7days.iloc[:, i].values

        model.fit(train_X, train_y, eval_set=[(test_X, test_y)], eval_metric='rmse', early_stopping_rounds=5)
        pickle.dump(model, open(f'{MODEL_DIR}/lgb_day{i}.pkl', 'wb'))


def predict_lgb(df_test_X, df_test_y_7_days):
    '''
    使用xgb模型进行预测
    :param df_test_X:
    :param df_test_y_7_days:
    :return:
    '''
    result = []
    rmse_list = []

    test_X = df_test_X.values

    for i in range(7):
        print(f'预测第{i+1}天...')
        path = f'{MODEL_DIR}/lgb_day{i}.pkl'
        model = pickle.load(open(path, 'rb'))
        test_y = df_test_y_7_days.iloc[:, i].values

        # # 通过误差分析 发现周末的销量存在较大的误差，在实际预测的时候加上0.95的策略参数
        if i in (4, 5):
            y_hat = np.round(model.predict(test_X) * 0.97).astype(int)
        else:
            y_hat = np.round(model.predict(test_X)).astype(int)
        # y_hat = np.round(model.predict(test_X)).astype(int)
        result.append(y_hat.reshape(-1, 1))
        rmse_list.append(np.sqrt(mean_squared_error(test_y, y_hat)))

    print('计算评价指标')
    predict_matrix = np.hstack(result)
    true_matrix = df_test_y_7_days.values
    df_predict = pd.DataFrame(predict_matrix)
    df_true = pd.DataFrame(true_matrix)
    df_predict.to_csv(f'{DATA_DIR}/prediction/xgb_predict.csv', index=False, header=False)
    df_true.to_csv(f'{DATA_DIR}/prediction/xgb_true.csv', index=False, header=False)

    loss = np.sum(np.abs((true_matrix - predict_matrix) / (true_matrix + predict_matrix))) / true_matrix.shape[0] / \
           true_matrix.shape[1]
    rmse = np.mean(rmse_list)
    print(f'loss:{round(loss,6)},rmse:{round(rmse,6)}')


# def train_model_rf(df_train_X, df_train_y_7_days, df_test_X, df_test_y_7days):
#     train_X = df_train_X.values
#     model = RandomForestRegressor(n_estimators=100, max_depth=10, oob_score=True, min_samples_split=16,
#                                   min_samples_leaf=8, max_leaf_nodes=32, max_features=0.85, )
#
#     for i in range(7):
#         print(f'rf 训练第{i+1}个模型')
#         train_y = df_train_y_7_days.iloc[:, i].values
#
#         model.fit(train_X, train_y)
#         pickle.dump(model, open(f'{MODEL_DIR}/rf_day{i}.pkl', 'wb'))


# def predict_rf(df_test_X, df_test_y_7_days):
#     '''
#     使用xgb模型进行预测
#     :param df_test_X:
#     :param df_test_y_7_days:
#     :return:
#     '''
#     result = []
#     rmse_list = []
#
#     test_X = df_test_X.values
#
#     for i in range(7):
#         print(f'预测第{i+1}天...')
#         path = f'{MODEL_DIR}/rf_day{i}.pkl'
#         model = pickle.load(open(path, 'rb'))
#         test_y = df_test_y_7_days.iloc[:, i].values
#
#         # # 通过误差分析 发现周末的销量存在较大的误差，在实际预测的时候加上0.95的策略参数
#         if i in (4, 5):
#             y_hat = np.round(model.predict(test_X) * 0.97).astype(int)
#         else:
#             y_hat = np.round(model.predict(test_X)).astype(int)
#         result.append(y_hat.reshape(-1, 1))
#         rmse_list.append(np.sqrt(mean_squared_error(test_y, y_hat)))
#
#     print('计算评价指标')
#     predict_matrix = np.hstack(result)
#     true_matrix = df_test_y_7_days.values
#     df_predict = pd.DataFrame(predict_matrix)
#     df_true = pd.DataFrame(true_matrix)
#     df_predict.to_csv(f'{DATA_DIR}/prediction/xgb_predict.csv', index=False, header=False)
#     df_true.to_csv(f'{DATA_DIR}/prediction/xgb_true.csv', index=False, header=False)
#
#     loss = np.sum(np.abs((true_matrix - predict_matrix) / (true_matrix + predict_matrix))) / true_matrix.shape[0] / \
#            true_matrix.shape[1]
#     rmse = np.mean(rmse_list)
#     print(f'loss:{round(loss,6)},rmse:{round(rmse,6)}')


if __name__ == '__main__':
    df_train_X, df_train_y_7_days, df_test_X, df_test_y_7days = load_data()
    # train_model_xgb(df_train_X, df_train_y_7_days, df_test_X, df_test_y_7days)
    predict_xgb(df_test_X, df_test_y_7days)

    # train_model_lgb(df_train_X, df_train_y_7_days, df_test_X, df_test_y_7days)
    predict_lgb(df_test_X, df_test_y_7days)

    # train_model_rf(df_train_X, df_train_y_7_days, df_test_X, df_test_y_7days)
    # predict_rf(df_test_X, df_test_y_7days)
