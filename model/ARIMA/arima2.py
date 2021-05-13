# encoding=utf8
import os
import pandas as pd
import matplotlib
import matplotlib.pylab as plt
import numpy as np
import sklearn
import sys
from datetime import date
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA, ARIMAResults
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from matplotlib.pylab import rcParams
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib import pyplot

matplotlib.use('agg')


# 检验统计稳定性
def test_stationarity(timeseries, window=6):
    # Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=window)
    rolstd = pd.rolling_std(timeseries, window=window)

    # Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # # Perform Dickey-Fuller test:
    # print('Results of Dickey-Fuller Test:')
    # dftest = adfuller(timeseries, autolag='AIC')
    # dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    # for key, value in dftest[4].items():
    #     dfoutput['Critical Value (%s)' % key] = value
    # print(dfoutput)


def acf_pacf_plot(ts_log_diff):
    plot_acf(ts_log_diff, lags=40)  # ARIMA,q
    plot_pacf(ts_log_diff, lags=40)  # ARIMA,p
    pyplot.show()


# 注意这里面使用的ts_log_diff是经过合适阶数的差分之后的数据，
# 上文中提到ARIMA该开源库，不支持3阶以上的#差分。所以我们需要提前将数据差分好再传入
# 求解最佳模型参数p,q
def _proper_model(ts_log_diff, maxLag):
    best_p = 0
    best_q = 0
    best_bic = sys.maxsize
    best_model = None
    for p in np.arange(maxLag):
        for q in np.arange(maxLag):
            try:
                model = ARMA(ts_log_diff, order=(p, q))
                results_ARMA = model.fit(disp=-1)
            except:
                continue
            bic = results_ARMA.bic
            # print(bic, best_bic)
            if bic < best_bic:
                best_p = p
                best_q = q
                best_bic = bic
                best_model = results_ARMA
    return best_p, best_q, best_model


print(os.getcwd())

timeseries = [Sin, Sout, Cin, Cout, Itra]
timeseriesname = ['Sin', 'Sout', 'Cin', 'Cout', 'Itra']


def pre_data(timeseries_fea, feaID, GridID):
    """
    prepare timeSeries and timeSeeries_diff
    """
    ts = pd.Series(timeseries_fea[GridID])  # [8784]
    ts_diff = ts - ts.shift()
    ts_diff.dropna(inplace=True)
    X = ts_diff.values
    ts_diff = X.astype('float64')  # numpy array
    X = ts.values
    ts = X.astype('float64')  # numpy array

    y_true = data_test[:, GridID, feaID]  # [144]

    return ts, ts_diff, y_true


# 验证一阶差分稳定性
# # 一阶差分
# ts_log_diff = ts_log - ts_log.shift()
# ts_log_diff.dropna(inplace=True)
# X = ts_log_diff.values
# ts_log_diff = X.astype('float32')


# test_stationarity(ts)
# test_stationarity(ts_log)
# test_stationarity(ts_log_diff)
# test_stationarity(ts_diff)

# acf_pacf_plot(ts_log_diff)
# acf_pacf_plot(ts_log_diff)

# monkey patch around bug in ARIMA class for save model
def __getnewargs__(self):
    return ((self.endog), (self.k_lags, self.k_diff, self.k_ma))


# train model and save model
def train_save_arima(ts, save_path, best_p, best_q):
    a = 0
    try:
        model = ARIMA(ts, order=(best_p, 1, best_q)) # 一阶差分
        results_ARIMA = model.fit(disp=-1)
        # Save Models
        ARIMA.__getnewargs__ = __getnewargs__
        # save model
        results_ARIMA.save(save_path)
    except:
        model = ARIMA(ts, order=(best_p, 1, best_q)) # 一阶差分
        results_ARIMA = model.fit(transparams=False, disp=-1)
        # Save Models
        ARIMA.__getnewargs__ = __getnewargs__
        # save model
        results_ARIMA.save(save_path)
        a = a+1
        print('pass')

    return results_ARIMA,a


# predict and eval model
def pred_eval_model(results_ARIMA, ts, forecast_n, y_true, fig_save_path):
    # forecast方法会自动进行差分还原，当然仅限于支持的1阶和2阶差分
    # forecast_n = 144  # 预测未来12个月走势
    forecast_ARIMA_log = results_ARIMA.forecast(forecast_n)
    forecast_ARIMA_log = forecast_ARIMA_log[0]
    # print(forecast_ARIMA_log[:144])

    # ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
    forecast_ARIMA_log = pd.Series(forecast_ARIMA_log,
                                   index=np.arange(len(ts[-2016:]) + 1, len(ts[-2016:]) + len(forecast_ARIMA_log) + 1,
                                                   1))
    forecast_ARIMA_log.fillna(forecast_ARIMA_log.mean(), inplace=True)

    MSE = sklearn.metrics.mean_squared_error(y_true, forecast_ARIMA_log)
    # diff = y_true - forecast_ARIMA_log  # [144]
    y_true_mean = np.mean(y_true)
    acc = MSE / y_true_mean

    y_true = pd.Series(y_true, index=np.arange(2016 + 1, 2016 + 144 + 1, 1))

    plt.plot(ts[-2016:], color="blue", label='Original')
    plt.plot(y_true, color="navy", label='y_true')
    plt.plot(forecast_ARIMA_log, color='red', label='Predicted')
    plt.legend(loc='best')
    plt.title('ARIMA MSE: %.4f ACC: %.4f' % (MSE, acc))
    plt.xlim([0, 2016+144])
    # show the biggest figure
    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()
    fig = plt.gcf()
    #plt.show()
    fig.savefig(fig_save_path, bbox_inches='tight',dpi=100)
    plt.close(fig)

    return MSE, acc



# predict model
for feaID in range(1,len(timeseriesname)):
    filename = open('./log_test_%s.log'%timeseriesname[feaID],'w')
    MSE_arr = []
    acc_arr = []
    for gridID in range(0,len(Sin),250):
        # Prepare train data and test data
        ts, ts_diff, y_true = pre_data(timeseries[feaID], feaID, gridID)

        # optimize the best model parameters
        best_p, best_q, model_ama = _proper_model(ts_diff, 10)  # 对一阶差分求最优p和q
        print(best_p, best_q)
        filename.write('Grid %d predict stat Info:\n'%gridID)
        filename.write('Best_p:%s, Best_q:%s\n'%(str(best_p),str(best_q)))

        # train and save model
        save_path = './train_model_arima_save/train_model_save_%s/ARIMA_model_%s_grid%d.pkl' % (
            timeseriesname[feaID], timeseriesname[feaID], gridID)
        results_ARIMA, a  = train_save_arima(ts, save_path, best_p, best_q)
        if a == 1:
            filename.write('###########################################parameters not converge very weill. \n')

        fig_save_path = './predict_figure_save/predict_figure_save_%s/ARIMA_fig_%s_grid%d.png' % (timeseriesname[feaID], timeseriesname[feaID], gridID)
        MSE, acc = pred_eval_model(results_ARIMA,ts=ts, forecast_n=144,y_true=y_true,fig_save_path=fig_save_path)
        filename.write('MSE: %s, acc: %s\n'%(str(MSE),str(acc)))
        MSE_arr.append(MSE)
        acc_arr.append(acc)

    MSE_mean = np.mean(MSE_arr)
    acc_mean = np.mean(acc_arr)
    print(MSE_mean)
    filename.write('MSE_mean: %s, acc_mean: %s\n'%(str(MSE_mean),str(acc_mean)))
    filename.close()
    MSE_ts = pd.Series(MSE_arr)
    acc_ts = pd.Series(acc_arr)
    plt.plot(MSE_ts, color="blue", label='MSE_timeSeries')
    plt.plot(acc_ts, color='red', label='Acc_timeSeries')
    plt.legend(loc='best')
    plt.title('%s MSE_mean: %s, acc_mean: %s\n'%(timeseriesname[feaID],str(MSE_mean),str(acc_mean)))
    mse_acc_save_path = './MSE_ACC_%s.png'%timeseriesname[feaID]
    plt.xlim([0, len(MSE_arr)])
    # show the biggest figure
    # manager = plt.get_current_fig_manager()
    # manager.window.showMaximized()
    fig = plt.gcf()
    #plt.show()
    fig.savefig(mse_acc_save_path, bbox_inches='tight',dpi=100)
    plt.close(fig)










# # load model
# results_ARIMA = ARIMAResults.load('ARIMA_model.pkl')





# 将预测的结果与原始图像画在一张图片上。
# #定义获取连续时间，start是起始时间，limit是连续的天数,level可以是day,month,year
# import arrow
# def get_date_range(start, limit, level='month',format='YYYY-MM-DD'):
#     start = arrow.get(start, format)
#     result=(list(map(lambda dt: dt.format(format) , arrow.Arrow.range(level, start, limit=limit))))
#     dateparse2 = lambda dates:pd.datetime.strptime(dates,'%Y-%m-%d')
#     return map(dateparse2, result)


# # 预测从1961-01-01开始，也就是我们训练数据最后一个数据的后一个日期
# new_index = get_date_range('1961-01-01', forecast_n)
# forecast_ARIMA_log = pd.Series(forecast_ARIMA_log, copy=True, index=new_index)
# print(forecast_ARIMA_log.head())
# # 直接取指数，即可恢复至原数据
# forecast_ARIMA = np.exp(forecast_ARIMA_log)
# print(forecast_ARIMA)
# plt.plot(ts,label='Original',color='blue')
# plt.plot(forecast_ARIMA, label='Forcast',color='red')
# plt.legend(loc='best')
# plt.title('forecast')
# plt.show()