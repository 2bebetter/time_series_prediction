import joblib
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score

class SVRModel(object):
    def __init__(self, file, train_size, test_size, x_lag):
        self.ts, self.time_index = self.load_data(file)
        # self.sample, self.label = self.time_slice(self.time_index, self.ts, x_lag=x_lag)
        self.test_size = test_size
        self.train_size = train_size
        self.train = self.ts[:train_size]
        self.test = self.ts[train_size:train_size+test_size]

    @staticmethod
    def load_data(filename, time_index='Time', time_series='Inttraffic'):
        """
        load data from csv file
        :return:
        """
        df = pd.read_csv(filename, sep=',', index_col=time_index, header=0)
        df.index = pd.to_datetime(df.index)
        ts = df[time_series]
        ts = ts.astype('float32')
        return ts, df.index

    @staticmethod
    def time_slice(time_index, single, x_lag):
        sample = []
        label = []
        for k in range(len(time_index) - x_lag - 1):
            t = k + x_lag
            sample.append(single[k:t])
            label.append(single[t + 1])
        return sample, label

    def get_average(self):
        average = float(sum(self.train)) / len(self.train)
        return average

    def get_stddev(self):
        average = float(sum(self.train)) / len(self.train)
        # 方差
        total = 0
        for value in self.train:
            total += (value - average) ** 2

        stddev = math.sqrt(total / len(self.train))
        return stddev

    def z_score_normalize(self, data):
        average = float(sum(data)) / len(data)

        # 方差
        total = 0
        for value in data:
            total += (value - average) ** 2

        stddev = math.sqrt(total / len(data))

        # z-score标准化方法
        return [(x - average) / stddev for x in data]

    def phase_space_reconstruction(self, t=1, d=5):
        pass

    def build_train_model(self):
        pass


def Create_dataset(dataset, look_back):
    data_X, data_Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        data_X.append(a)
        data_Y.append(dataset[i + look_back])
    data_X = np.array(data_X)
    data_Y = np.array(data_Y)
    return data_X, data_Y


#单维最大最小归一化和反归一化函数
def Normalize(list):
    list = np.array(list)
    low, high = np.percentile(list, [0, 100])
    delta = high - low
    if delta != 0:
        for i in range(0, len(list)):
            list[i] = (list[i]-low)/delta
    return list, low, high


def FNoramlize(list, low, high):
    delta = high - low
    if delta != 0:
        for i in range(0, len(list)):
            list[i] = list[i]*delta + low
    return list


def Normalize2(list, low, high):
    list = np.array(list)
    delta = high - low
    if delta != 0:
        for i in range(0, len(list)):
            list[i] = (list[i]-low)/delta
    return list


def evaluate(filename):
    # 读取数据，选择6-7和6-8作为训练数据集，共576个点，6-9的数据作为测试集，共288个点
    model = SVRModel(file=filename, train_size=1068, test_size=864)
    # 对训练集进行归一化
    # std_sample = model.get_stddev(model.sample)
    # average_sample = model.get_average(model.sample)
    # sample = model.z_score_normalize(model.sample)
    #
    # std_label = model.get_stddev(model.label)
    # average_label = model.get_average(model.label)
    # label = model.z_score_normalize(model.label)
    # mm = MinMaxScaler()
    # sample = mm.fit_transform(model.sample)
    # X_train, X_test, y_train, y_test = train_test_split(model.train, model.label,
    #                                                     train_size=492, test_size=288,
    #                                                     random_state=42)

    # scale = StandardScaler()
    # scale_fit = scale.fit(X_train)
    # X_train = scale_fit.transform(X_train)
    #
    # scale = StandardScaler()
    # scale_fit = scale.fit(X_test)
    # X_test = scale_fit.transform(X_test)

    # 新的归一化方法
    train_n, train_low, train_high = Normalize(model.train)
    test_n = Normalize2(model.test, train_low, train_high)
    print(train_n, test_n)

    train_X, train_Y = Create_dataset(train_n, look_back=1)
    test_X, test_Y = Create_dataset(test_n, look_back=1)
    # 额外添加一个维度使train_X，test_X变为三维
    print(train_X.shape)
    # train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
    # test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

    parameters = {'kernel': ['rbf'], 'gamma': np.logspace(-5, 0, num=6, base=2.0),
                  'C': np.logspace(-5, 5, num=11, base=2.0)}

    # 网格搜索：选择十折交叉验证
    svr = SVR()
    grid_search = GridSearchCV(svr, parameters, cv=10, n_jobs=4, scoring='neg_mean_squared_error')
    # SVR模型训练
    grid_search.fit(train_X, train_Y)
    # 输出最终的参数
    print(grid_search.best_params_)

    # 模型的精度
    print(grid_search.best_score_)

    # SVR模型保存
    joblib.dump(grid_search, 'svr.pkl')

    # SVR模型加载
    svr = joblib.load('svr.pkl')

    # SVR模型测试
    train_predict = svr.predict(train_X)
    test_predict = svr.predict(test_X)

    # 反归一化
    train_Y = FNoramlize(train_Y, train_low, train_high)
    train_predict = FNoramlize(train_predict, train_low, train_high)
    test_Y = FNoramlize(test_Y, train_low, train_high)
    test_predict = FNoramlize(test_predict, train_low, train_high)

    # 进行绘图
    plt.subplot(121)
    plt.plot(train_Y)
    plt.plot(train_predict)
    plt.subplot(122)
    plt.plot(test_Y)
    plt.plot(test_predict)
    plt.show()

    print('RMSE: %.4f' % np.sqrt(sum((test_predict.values - test.values) ** 2) / test.size))


def RMSE(actual, predicted):
    """
    Equation for root mean square error
    :param actual:
    :param predicted:
    :return:
    """
    summedVal = 0
    for k, element in enumerate(predicted):
        val_pred, val_act = element, actual[k]
        summedVal += (val_pred - val_act) ** 2

    N = len(predicted)
    RMSE = (summedVal / N) ** (1 / 2)
    return RMSE


if __name__ == '__main__':
    filename = "dataset/inttraffic.csv"
    evaluate(filename)
