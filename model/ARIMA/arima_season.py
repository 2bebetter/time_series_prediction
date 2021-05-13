# encoding=utf8

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller as ADF
import statsmodels.tsa.stattools as st
import matplotlib.pyplot as plt
import warnings


class ArimaSeasonModel(object):
    def __init__(self, file, train_size, test_size):
        self.ts = self.load_data(file)
        self.test_size = test_size
        self.train_size = train_size
        self.train = self.ts[:train_size]
        self.test = self.ts[train_size:train_size+test_size]

    @staticmethod
    def load_data(filename):
        """
        load data from csv file
        :return:
        """
        df = pd.read_csv(filename, sep=',', index_col='Time', header=0)
        df.index = pd.to_datetime(df.index)
        ts = df['Inttraffic']
        return ts

    def decompose(self, freq):
        """
        对时间序列进行分解
        :param freq: 周期，由于每隔5分钟一次采样，所以周期为60*24/5=288
        :return:
        """
        decomposition = seasonal_decompose(self.train, freq=freq, two_sided=False)
        self.trend = decomposition.trend
        self.seasonal = decomposition.seasonal
        self.residual = decomposition.resid
        self.trend.dropna(inplace=True)
        self.seasonal.dropna(inplace=True)
        self.residual.dropna(inplace=True)

        d = self.residual.describe()
        delta = d['75%'] - d['25%']

        self.low_error, self.high_error = (d['25%'] - 1 * delta, d['75%'] + 1 * delta)

    def trend_model(self, order):
        """
        为分解出来的趋势数据单独建模
        :param order:
        :return:
        """
        self.trend.dropna(inplace=True)
        self.trend_model = ARIMA(self.trend, order=order).fit()

        return self.trend_model

    def add_season(self):
        """
        为预测出的趋势数据添加周期数据和残差数据
        :return:
        """
        self.train_season = self.seasonal
        values = []
        low_conf_values = []
        high_conf_values = []

        for i, t in enumerate(self.pred_time_index):
            trend_part = self.trend_pred[i]

            # 相同时间的数据均值
            season_part = self.train_season[
                self.train_season.index.time == t.time()
                ].mean()

            # 趋势+周期+误差界限
            predict = trend_part + season_part
            low_bound = trend_part + season_part + self.low_error
            high_bound = trend_part + season_part + self.high_error

            values.append(predict)
            low_conf_values.append(low_bound)
            high_conf_values.append(high_bound)

        self.final_pred = pd.Series(values, index=self.pred_time_index, name='predict')
        self.low_conf = pd.Series(low_conf_values, index=self.pred_time_index, name='low_conf')
        self.high_conf = pd.Series(high_conf_values, index=self.pred_time_index, name='high_conf')

    def proper_fit(self, ts, min_lag=0, max_lag=6):
        warnings.filterwarnings("ignore")
        results_list = []
        for p in np.arange(min_lag, max_lag):
            for d in np.arange(3):
                for q in np.arange(min_lag, max_lag):
                    try:
                        param = (p, d, q)
                        mod = ARIMA(ts, order=(p, d, q))
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

    def predict_new(self):
        '''
        预测新数据
        '''
        # 续接train，生成长度为n的时间索引，赋给预测序列
        n = self.test_size
        self.pred_time_index = pd.date_range(start=self.train.index[-1], periods=n + 1, freq='5min')[1:]
        self.trend_pred = self.trend_model.forecast(n)[0]
        self.add_season()


def evaluate(filename):
    #model = ArimaSeasonModel(file=filename, train_size=780, test_size=288)
    model = ArimaSeasonModel(file=filename, train_size=1068, test_size=864)
    model.decompose(freq=288)
    #best_p, best_d, best_q = model.proper_fit(model.trend)
    # ARIMA(2, 2, 5) with lowest_AIC:14117.954336624123
    model.trend_model(order=(2, 2, 5))
    model.predict_new()
    pred = model.final_pred
    test = model.test

    plt.subplot(211)
    plt.plot(model.ts)
    plt.title(filename.split('.')[0])
    plt.subplot(212)
    pred.plot(color='salmon', label='Predict')
    test.plot(color='steelblue', label='Original')
    #model.low_conf.plot(color='grey', label='low')
    #model.high_conf.plot(color='grey', label='high')

    plt.legend(loc='best')
    plt.title('RMSE: %.4f' % np.sqrt(sum((pred.values - test.values) ** 2) / test.size))
    plt.tight_layout()
    plt.show()

    print(test.values)

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
