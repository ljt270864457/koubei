import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import xgboost

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')

result = []
rmse_list = []


def train_model():
    df_train1_sample = pd.read_csv(f'{DATA_DIR}/sample_train1_2016-09-20_2016-10-24.csv', skiprows=1, header=None)
    df_train2_sample = pd.read_csv(f'{DATA_DIR}/sample_train2_2016-09-13_2016-10-17.csv', skiprows=1, header=None)
    df_train3_sample = pd.read_csv(f'{DATA_DIR}/sample_train3_2016-09-06_2016-10-10.csv', skiprows=1, header=None)

    df_train1_label = pd.read_csv(f'{DATA_DIR}/label_train1_2016-09-20_2016-10-24.csv', skiprows=1, header=None)
    df_train2_label = pd.read_csv(f'{DATA_DIR}/label_train2_2016-09-13_2016-10-17.csv', skiprows=1, header=None)
    df_train3_label = pd.read_csv(f'{DATA_DIR}/label_train3_2016-09-06_2016-10-10.csv', skiprows=1, header=None)

    df_train_X = pd.concat([df_train1_sample, df_train2_sample, df_train3_sample])
    df_train_y_7_days = pd.concat([df_train1_label, df_train2_label, df_train3_label])

    model = xgboost.XGBRegressor()
    train_X = df_train_X.values

    for i in range(7):
        print(f'训练第{i+1}个模型，预测第{i+1}天的数据....')
        train_y = df_train_y_7_days.iloc[:, i].values

        model.fit(train_X, train_y)
        pickle.dump(model, open(f'{MODEL_DIR}/xgb_day{i}.pkl', 'wb'))


def predict():
    result = []
    rmse_list = []

    df_test_X = pd.read_csv(f'{DATA_DIR}/sample_test_2016-09-27_2016-10-31.csv', skiprows=1, header=None)
    df_test_y_7_days = pd.read_csv(f'{DATA_DIR}/label_test_2016-09-27_2016-10-31.csv', skiprows=1, header=None)

    test_X = df_test_X.values

    for i in range(7):
        print(f'预测第{i+1}天...')
        path = f'{MODEL_DIR}/xgb_day{i}.pkl'
        model = pickle.load(open(path, 'rb'))
        test_y = df_test_y_7_days.iloc[:, i].values
        y_hat = model.predict(test_X)
        result.append(y_hat.reshape(-1, 1))
        rmse_list.append(np.sqrt(mean_squared_error(test_y, y_hat)))

    print('计算评价指标')
    predict_matrix = np.hstack(result)
    true_matrix = df_test_y_7_days.values
    loss = np.sum(np.abs((true_matrix - predict_matrix) / (true_matrix + predict_matrix))) / true_matrix.shape[0] / \
           true_matrix.shape[1]
    rmse = np.mean(rmse_list)
    print(f'loss:{round(loss,6)},rmse:{round(rmse,6)}')


if __name__ == '__main__':
    train_model()
    predict()
