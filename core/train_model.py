import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import xgboost


def eval(y_true, y_hat):
    '''
    评估指标
    :param y_true:
    :param y_hat:
    :return:
    '''
    y_true = y_true.reshape((-1, 1))
    y_hat = y_hat.reshape((-1, 1))
    matrix = np.hstack([y_true, y_hat])
    return np.sum((matrix[:, 0] - matrix[:, 1]) / (matrix[:, 0] + matrix[:, 1])) / matrix.shape[0]


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

df_train1 = pd.read_csv(f'{DATA_DIR}/train1_2016-09-20_2016-10-24.csv', header=None)
df_train2 = pd.read_csv(f'{DATA_DIR}/train2_2016-09-10_2016-10-14.csv', header=None)
df_train3 = pd.read_csv(f'{DATA_DIR}/train3_2016-09-03_2016-09-30.csv', header=None)
df_train = pd.concat([df_train1, df_train2, df_train3])
df_train_X = df_train.iloc[:, 1:-1]
df_train_y = df_train.iloc[:, -1]
print(df_train_X)
print(df_train_y)

model = xgboost.XGBRegressor()
model.fit(df_train_X.values, df_train_y.values)

df_test = pd.read_csv(f'{DATA_DIR}/test_2016-09-27_2016-10-31.csv', header=None)
print(df_train.head())
df_test_X = df_test.iloc[:, 1:-1]
df_test_y = df_test.iloc[:, -1]

print(123)
y_hat = model.predict(df_test_X.values)
df = pd.DataFrame(zip(y_hat, df_test_y.values), columns=['predict', 'true'])
print(df.head())
print(np.sqrt(mean_squared_error(df_test_y.values, y_hat)))
# print(eval(df_test_y.values, y_hat))
