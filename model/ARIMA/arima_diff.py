# encoding=utf8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
import warnings
from statsmodels.tsa.arima_model import ARIMA


class ArimaModel(object):
    def __init__(self, file, test_size):
        self.ts = self.load_data(file)
        self.test_size = test_size
        self.train_size = len(self.ts) - self.test_size
        self.train = self.ts[:len(self.ts)-test_size]
        self.test = self.ts[-test_size:]

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
        return ts

    def build_train_model(self, best_p, diff, best_q):
        """
        :param best_p:
        :param diff:
        :param best_q:
        :return:
        """
        self.train.dropna(inplace=True)
        order = (best_p, diff, best_q)
        self.train_model = ARIMA(self.train, order).fit(disp=-1, method='css')
        return self.train_model

    def predict_new(self):
        """
        预测新数据
        :return:
        """
        # 续接train，生成长度为n的时间索引，赋给预测序列
        n = self.test_size
        self.pred_time_index = pd.date_range(start=self.train.index[-1], periods=n + 1, freq='5min')[1:]
        self.train_pred = self.train_model.forecast(n)[0]
        return self.train_pred

    def proper_model(self, ts_log_diff, max_lag=10):
        best_p = 0
        best_q = 0
        best_bic = sys.maxsize
        best_model = None
        for p in np.arange(max_lag):
            for q in np.arange(max_lag):
                try:
                    model = ARIMA(ts_log_diff, order=(p, 1, q))
                    results_ARIMA = model.fit(disp=-1)
                except Exception as e:
                    print(e)
                    continue
                bic = results_ARIMA.bic
                print(p, q, bic, best_bic)
                # print(bic, best_bic)
                if bic < best_bic:
                    best_p = p
                    best_q = q
                    best_bic = bic
                    best_model = results_ARIMA
        self.train_model = best_model
        return best_p, best_q, best_model

    def proper_fit(self, ts, min_lag=9, max_lag=12):
        warnings.filterwarnings("ignore")
        results_list = []
        for p in np.arange(15, 20):
            for q in np.arange(12, 15):
                try:
                    param = (p, 0, q)
                    mod = ARIMA(ts, order=(p, 0, q))
                    results = mod.fit()

                    print('ARIMA{} - AIC:{}'.format(param, results.aic))
                    results_list.append([param, results.aic])
                except Exception as e:
                    print('Exception:\n', e)
                    continue
        results_list = np.array(results_list)
        lowest_AIC = np.argmin(results_list[:, 1])
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('ARIMA{} with lowest_AIC:{}'.format(
            results_list[lowest_AIC, 0], results_list[lowest_AIC, 1]))
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        mod = ARIMA(ts, order=results_list[lowest_AIC, 0])
        self.train_model = mod.fit()
        return results_list[lowest_AIC, 0]


def evaluate(filename):
    model = ArimaModel(file=filename, test_size=14)
    best_p, best_d, best_q = model.proper_fit(model.train, 10)
    print(best_p, best_d, best_q)
    # (9, 0, 9)
    # best_p = 20
    # best_d = 0
    # best_q = 11
    model.build_train_model(best_p, best_d, best_q)
    pred = model.predict_new()

    # ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
    pred = pd.Series(pred, index=model.pred_time_index)
    pred.fillna(pred.mean(), inplace=True)

    test = model.test

    plt.subplot(211)
    plt.plot(model.ts)
    plt.title(filename.split('.')[0])
    plt.subplot(212)
    pred.plot(color='salmon', label='Predict')
    test.plot(color='steelblue', label='Original')
    # model.low_conf.plot(color='grey', label='low')
    # model.high_conf.plot(color='grey', label='high')

    plt.legend(loc='best')
    plt.title('RMSE: %.4f' % np.sqrt(sum((pred.values - test.values) ** 2) / test.size))
    plt.tight_layout()
    plt.show()

    print(pred.values)

    print('RMSE: %.4f' % np.sqrt(sum((pred.values - test.values) ** 2) / test.size))


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
    RMSE = (summedVal / N) ** (1/2)
    return RMSE


if __name__ == '__main__':
    filename = "inttraffic.csv"
    evaluate(filename)
