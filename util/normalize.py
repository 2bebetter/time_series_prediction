import math


def minmax_normalize(data):
    res = [(x - min(data)) / (max(data) - min(data)) for x in data]
    return res


def z_score_normalize(data):
    average = float(sum(data)) / len(data)

    # 方差
    total = 0
    for value in data:
        total += (value - average) ** 2

    stddev = math.sqrt(total / len(data))

    # z-score标准化方法
    res = [(x - average) / stddev for x in data]
    return res


def mean_normalize(data, type='max'):
    average = float(sum(data)) / len(data)

    # 均值归一化方法
    if type == 'max':
        # max为分母
        res = [(x - average) / max(data) for x in data]
        return res
    else:
        # max-min为分母
        res = [(x - average) / (max(data) - min(data)) for x in data]
        return res


def log2_normalize(data):
    # log2函数转换
    res = [math.log2(x) for x in data]
    return res


def log10_normalize(data):
    # log10函数转换
    res = [math.log10(x) for x in data]
    return res


def atan_normalize(data):
    res = [math.atan(x) for x in data]
    return res
