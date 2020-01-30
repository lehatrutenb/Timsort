import json
import os
from Timsort_lib import *
from keras import models


def timsort(arr: list):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    model = models.load_model('network.h5')
    with open('data.json') as f_read:
        data = json.load(f_read)
    mean, std = data.values()
    arr_len = len(arr)
    arr_len -= mean
    arr_len /= std
    minrun = int(model.predict([arr_len])[0])
    return TimSort(arr, minrun)


def fast_timsort(arr: list, model):
    with open('data.json') as f_read:
        data = json.load(f_read)
    mean, std = data.values()
    arr_len = len(arr)
    arr_len -= mean
    arr_len /= std
    minrun = int(model.predict([arr_len])[0])
    return TimSort(arr, minrun)


def prime_fast():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    return models.load_model('network.h5')
