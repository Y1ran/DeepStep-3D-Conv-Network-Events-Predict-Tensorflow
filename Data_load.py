# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 21:44:15 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('yellow_tripdata_2016-05.csv')
df.head(10)

# 对数据集进行清洗，去除异常值，只保留相关信息。
def data_cleaning(df):
    # 只留下相关的列
    new_df = df.iloc[:,[1,2,3,4,5,6,9,10]]
    
    # 过滤掉经纬度超出城市范围的行程和经纬度异常的行程
    new_df = new_df[(-74.259090<=new_df['pickup_longitude']) & (new_df['pickup_longitude']<=-73.700181)]
    new_df = new_df[(-74.259090<=new_df['dropoff_longitude']) & (new_df['dropoff_longitude']<=-73.700181)]
    new_df = new_df[(40.477399<=new_df['pickup_latitude']) & (new_df['pickup_latitude']<=40.916179)]
    new_df = new_df[(40.477399<=new_df['dropoff_latitude']) & (new_df['dropoff_latitude']<=40.916179)]
    
    # 过滤掉行程距离异常的行程
    new_df = new_df[(0.1<=new_df['trip_distance']) & (new_df['trip_distance']<50)]
    
    # 过滤掉没有乘客的行程
    new_df = new_df[new_df['passenger_count']>0] 
    
    # 把行程时间由str转换为时间戳
    new_df['tpep_pickup_datetime'] = pd.to_datetime(new_df['tpep_pickup_datetime'])
    new_df['tpep_dropoff_datetime'] = pd.to_datetime(new_df['tpep_dropoff_datetime'])
    
    # 过滤下车时间没有晚于上车时间的行程
    new_df = new_df[new_df['tpep_dropoff_datetime'] > new_df['tpep_pickup_datetime']]
        
    # 把行程按照上车时间先后排序
    new_df = new_df.sort_values(by='tpep_pickup_datetime')

    return new_df

new_df = data_cleaning(df)
rng = pd.date_range('2016-05-01', '2016-06-01', freq='30T')  # 创建5月份30分钟为间隔的时间戳

intervals = []
for t in rng[1:]:
    intervals.append(t)
    
new_df.head()

# 增加一列，用来编号时间区间，时间间隔30分钟，第0个区间为[2016-05-01 00:00:00, 2016-05-01 00:30:00], 
new_df['pickup_time_interval'] = 0 

for idx, t in enumerate(intervals):
    if idx == 0:
        print('test')
        pass
    else:
        bool_index = (new_df['tpep_pickup_datetime']>intervals[idx-1]) & (new_df['tpep_pickup_datetime']<t) # 选定在30分钟区间内的行程
        new_df['pickup_time_interval'][bool_index] = idx  # 对这些行程编号为第i个区间
        
new_df['dropoff_time_interval'] = 1488  # 1489用来统一标记下车时间已是6月1日的区间。

for idx, t in enumerate(intervals):
    if idx == 0:
        print('test')
        new_df['dropoff_time_interval'][new_df['tpep_dropoff_datetime'] < t] = 0
    else:
        bool_index = (new_df['tpep_dropoff_datetime']>intervals[idx-1]) & (new_df['tpep_dropoff_datetime']<t) # 选定在30分钟区间内的行程
        new_df['dropoff_time_interval'][bool_index] = idx  # 对这些行程编号为第i个区间
        
new_df.head()

long_interval = np.linspace(-74.259090, -73.700181, num=64) # 生成固定间隔的纬度区间

new_df['pickup_longitude_interval'] = 0  # 标记纬度的区间，划分为0-63，共64组。

for idx, t in enumerate(long_interval):
    if idx == 63:
        continue
    bool_index = (new_df['pickup_longitude']>t) & (new_df['pickup_longitude']<=long_interval[idx+1]) # 选定上车位置在(t, t+1]纬度区间内的行程
    new_df['pickup_longitude_interval'][bool_index] = idx  # 对这些行程编号为第idx个区间


new_df['dropoff_longitude_interval'] = 0  # 标记纬度的区间，划分为0-63，共64组。

for idx, t in enumerate(long_interval):
    if idx == 63:
        continue
    bool_index = (new_df['dropoff_longitude']>t) & (new_df['dropoff_longitude']<=long_interval[idx+1]) # 选定下车位置在(t, t+1]纬度区间内的行程
    new_df['dropoff_longitude_interval'][bool_index] = idx  # 对这些行程编号为第idx个区间
lati_interval = np.linspace(40.477399, 40.916179, num=64) # 生成固定间隔的经度区间
new_df[new_df['dropoff_longitude_interval'] == 0]

new_df['pickup_latitude_interval'] = 0  # 标记经度的区间，划分为0-63，共64组。

for idx, t in enumerate(lati_interval):
    if idx == 63:
        continue
    bool_index = (new_df['pickup_latitude']>t) & (new_df['pickup_latitude']<=lati_interval[idx+1]) # 选定上车位置在(t, t+1]经度区间内的行程
    new_df['pickup_latitude_interval'][bool_index] = idx  # 对这些行程编号为第idx个区间
    

new_df['dropoff_latitude_interval'] = 0  # 标记经度的区间，划分为0-63，共64组。

for idx, t in enumerate(lati_interval):
    if idx == 63:
        continue
    bool_index = (new_df['dropoff_latitude']>t) & (new_df['dropoff_latitude']<=lati_interval[idx+1]) # 选定下车位置在(t, t+1]经度区间内的行程
    new_df['dropoff_latitude_interval'][bool_index] = idx  # 对这些行程编号为第idx个区间

# 至此时空区间的编码都完成，先去除多余的列。

result_df = new_df.iloc[:, [2, 8,9,10,11,12,13]]
result_df.head()
#result_df.to_csv('encoded-2016-05.csv')
result_df.groupby('pickup_time_interval', 'pickup_longitude_interval', 'pickup_latitude_interval').count()