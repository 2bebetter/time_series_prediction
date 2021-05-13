# encoding=utf8

import numpy as np
import pandas as pd
from util import timeseries_plot, config_plot
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller as ADF
import statsmodels.tsa.stattools as st
import matplotlib.pyplot as plt


config_plot()


def load_data(filename, split):
    """
    load data from csv file
    :return:
    """
    df = pd.read_csv(filename, sep=',', index_col='Time', header=0)
    df.index = pd.to_datetime(df.index)
    split_boundary = int(df.shape[0] * split)
    train_data = df[:split_boundary]
    print('train_x shape:', train_data.shape)

    # 构建测试集
    test_data = df[split_boundary:]
    print('test_x shape:', test_data.shape)
    return df, train_data, test_data


df, train_data, test_data = load_data("dataset/inttraffic.csv", 0.9)

train_data.plot(figsize=(15, 5))
plt.title('Internet traffic')
# 观察数据的趋势,发现是具有周期性的序列
plt.show()

# 2.1 检验平稳性

#plot_acf(train_data)
#plot_pacf(train_data)

adf_data = ADF(train_data)
print(adf_data)
# 结果：(-12.206674612928081, 1.189217226167012e-22, 41, 13252, {'1%': -3.4308435532092543,
# '5%': -2.8617581270545536, '10%': -2.5668861041328106}, 533092.8446687632)
# 1. 1%、%5、%10不同程度拒绝原假设的统计值和ADF Test result的比较，ADF Test result同时小于1%、5%、10%即说明非常好地拒绝该假设，本数据中，adf结果为-12， 小于三个level的统计值。
# 2. P-value是否非常接近0.本数据中，P-value 为 1e-22,接近0.


# 它是具有周期性的序列，但没办法使用seasonal_decompose分解，需要先提前知悉period的值，如何获取period？
decomposition = seasonal_decompose(train_data, model="additive", freq=288)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
fig = decomposition.plot()
plt.show()


# 有四种处理平稳性的方法
def draw_trend(timeSeries, size):
    """
    平滑法分为移动平均法和指数平滑法
    :param timeSeries:
    :param size:
    :return:
    """
    f = plt.figure(facecolor='white')

    # 对size个数据进行移动平均
    rol_mean = timeSeries.rolling(window=size).mean()

    # 对size个数据进行加权移动平均
    rol_weighted_mean = pd.DataFrame.ewm(timeSeries, span=size).mean()

    timeSeries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.show()


# 2.2 白噪声检验，白噪声检验也成为纯随机性检验。如果数据是纯随机性数据，那在进行数据分析就没有意义了
p_value = acorr_ljungbox(trend, lags=288, boxpierce=False, return_df=False)
print(p_value)
p_value = acorr_ljungbox(train_data, lags=[6, 12], boxpierce=False, return_df=False)
print(p_value)
# 结果： (array([ 78485.51727472, 153112.51773145]), array([0., 0.]))
# 第一个数组为延迟阶数的LB统计量的值
# 第二个数组为延迟阶数的P值
# 延迟6阶的P值为0, 所以认为该序列是非白噪声序列

order = st.arma_order_select_ic(trend, max_ar=2, max_ma=2, ic=['aic', 'bic'])
print(order.bic_min_order)
# (2, 2)

# 构造模型
model = ARMA(trend, order=(2, 2))
result_arma = model.fit()

predict_ts = result_arma.predict()

# 预测
d = residual.describe()
delta = d['75%'] - d['25%']
low_error, high_error = (d['25%'] - 1 * delta, d['75%'] + 1 * delta)


# 一阶差分还原
shift_ts = trend.shift(1)
recover_1 = predict_ts.add(shift_ts)

train_data.plot(figsize=(10, 5))
recover_1.plot(color='red')

ts = recover_1[~recover_1.isnull()]
df_fill = train_data[ts.index]
print('RMSE:{}'.format(np.sqrt(sum((ts-df_fill)**2)/ts.size)))
