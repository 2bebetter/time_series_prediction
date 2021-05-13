import pandas as pd
from model.ARIMA.arima import Arima_Class
from util import config_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

config_plot()


def load_data(filename, split):
    """
    load data from csv file
    :return:
    """
    df = pd.read_csv(filename, sep=',', index_col=0, header=0)
    split_boundary = int(df.shape[0] * split)
    train_data = df[:split_boundary]
    print('train_x shape:', train_data.shape)

    # 构建测试集
    test_data = df[split_boundary:]
    print('test_x shape:', test_data.shape)
    return train_data, test_data


train_data, test_data = load_data("dataset/inttraffic.csv", 0.9)

arima_para = {
    'p': range(2),
    'd': range(2),
    'q': range(2)
}
bucket_size = "30T"
seasonal_para = round(24 * 60 / (float(bucket_size[:-1])))
arima = Arima_Class(arima_para, seasonal_para)

# time series plot
ts_label = 'Internet Traffic'
plot_acf(train_data)
plot_pacf(train_data)
plt.title('acf and pacf')
plt.show()

arima.fit(train_data)

# Prediction on observed data starting on pred_start
# observed and prediction starting dates in plots
plot_start = '2005-06-07 07:00:00'
pred_start = '2005-07-28 13:55:00'

# One-step ahead forecasts
dynamic = False
arima.pred(train_data, plot_start, pred_start, dynamic, ts_label)

# Dynamic forecasts
dynamic = True
predict_ts = arima.pred(train_data, plot_start, pred_start, dynamic, ts_label)


# Forecasts to unseen future data
n_steps = 100  # next 100 * 30 min = 50 hours
arima.forcast(train_data, n_steps)