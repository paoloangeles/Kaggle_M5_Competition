#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Created on Mon May 4 19:51:01 2020

@author: paoloangeles
"""

# Import modules

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from itertools import cycle
from scipy import signal

pd.set_option('max_columns', 50)
plt.style.use('bmh')
color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

####### Reset figures #######
plt.close("all")

# Get data from pickle files

calendar = pd.read_pickle("./calendar.pkl")
sell_prices = pd.read_pickle("./sell_prices.pkl")
sales_train_validation = pd.read_pickle("./sales_train_validation.pkl")

####### Data preprocessing #######

## Check for any NaN values

total_num_days = 6
for day in range(total_num_days):
    null_rows = sales_train_validation["d_" + str(day+1)].isnull().any()
print(null_rows)

######## Data visualisation and exploration ########

# specify sales data columns

sales_columns = [column for column in sales_train_validation.columns if 'd_' in column]


### Merge calendar dates for one product ####
example = sales_train_validation.loc[sales_train_validation['id'] == 'HOBBIES_1_001_CA_1_validation'][sales_columns].T
example = example.rename(columns={8412:'HOBBIES_1_001_CA_1_validation'})
example = example.reset_index().rename(columns={'index': 'd'})
example = example.reset_index().rename(columns={0: 'sales'})
example = example.merge(calendar, how='left', validate='1:1')


# Split training data into historical and predicted data
x_train = example.loc[example.index[1800:-28], ['sales']].values.tolist()
y_train = example.loc[example.index[-28:], ['sales']].values.tolist()

# flatten training data lists by iterating through 2 for loops
x_train = [item for sublist in x_train for item in sublist]
y_train = [item for sublist in y_train for item in sublist]

plt.plot(x_train)

#### Data exploration ####

def Plot_item(product_id):

####### PLOT ALL DATA #######
    
    
    # select item, reindex to id and keep sales data columns, transform, plot
    
    sales_train_validation.loc[sales_train_validation['id'] == product_id, sales_columns].T.plot(figsize=(15, 5), title = 'HOBBIES_1_001_CA_1 sales by "d" number', color = next(color_cycle))
    plt.legend('')
    plt.show()
    
    
    # def Plot_range(product_id, start_index_sales, start_point):

Plot_item('HOBBIES_1_001_CA_1_validation')

def Plot_item_range(product_id, start_day, day_range, num_previous_ranges):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for i in range(-1*num_previous_ranges, 0):
    
        start_index = start_day + (day_range * i) - 1
        end_index = start_day + (day_range * (i + 1)) - 1
        
        x = list(range(day_range))
        y = sales_train_validation.loc[sales_train_validation['id'] == product_id, "d_" + str(start_index):"d_" + str(end_index-1)].T
    
        ax.plot(x, y)
        
    ax.set_xlabel('Day number')
    ax.set_ylabel('Number of sales of ' + str(product_id))
    
def Plot_item_previous_years(product_id = "", start_day = 1, day_range = 28, num_previous_years = 1):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for i in range(-1*num_previous_years, 0):
    
        start_index = start_day + (day_range * i) - 1
        end_index = start_day + (day_range * (i + 1)) - 1
        
        x = list(range(day_range))
        y = sales_train_validation.loc[sales_train_validation['id'] == product_id, "d_" + str(start_index):"d_" + str(end_index-1)].T
    
        ax.plot(x, y)
        
    ax.set_xlabel('Day number')
    ax.set_ylabel('Number of sales of ' + str(product_id))



####### Feature creation #######

def num_days_until_event(data):
    dataOut = np.empty(len(data))
    dataOut[:] = np.NaN
    eventOccur = np.array([i for i, val in enumerate(data) if val])
    eventOccur = eventOccur + 1
    eventOccur = np.insert(eventOccur, 0, 0)
    for event, nextEvent in zip(eventOccur[:-1], eventOccur[1:]):
        dataOut[event: nextEvent] = list(reversed(range(nextEvent-event)))
    return dataOut

# def num_days_after_event(data):
    

gaus = signal.gaussian(365, 100)
features = pd.DataFrame()

# Keep this column as key for any future merges/ reference
features['wm_yr_wk'] = example['wm_yr_wk']

# Weekend Proximity feature- registers 0 if day is weekend and incremented by 1 for each day away from weekend
features['weekend_prox'] = np.zeros(len(example.index))
features.loc[example['wday'] == 1, 'weekend_prox'] = 0
features.loc[example['wday'] == 2, 'weekend_prox'] = 0
features.loc[example['wday'] == 3, 'weekend_prox'] = 1
features.loc[example['wday'] == 4, 'weekend_prox'] = 2
features.loc[example['wday'] == 5, 'weekend_prox'] = 3
features.loc[example['wday'] == 6, 'weekend_prox'] = 2
features.loc[example['wday'] == 7, 'weekend_prox'] = 1

# Features for the days until the next of a type of event occur
sportOccur = np.logical_or(example['event_type_1']=='Sporting', example['event_type_2']=='Sporting')
features['days_until_sport'] = num_days_until_event(sportOccur)
cultureOccur = np.logical_or(example['event_type_1']=='Cultural', example['event_type_2']=='Cultural')
features['days_until_culture'] = num_days_until_event(cultureOccur)
religionOccur = np.logical_or(example['event_type_1']=='Religious', example['event_type_2']=='Religious')
features['days_until_religion'] = num_days_until_event(religionOccur)
nationalOccur = np.logical_or(example['event_type_1']=='National', example['event_type_2']=='National')
features['days_until_national'] = num_days_until_event(nationalOccur)



#### Plot real vs predicted values ####

def Create_time_steps(length_data, direction):
    if direction == 'backward':
        return list(range(-length_data, 0))
    elif direction == 'forward':
        return list(range(length_data))

def Plot_realvpred(plot_data, product_id):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'r.-', 'g.-']
    time_steps_x = Create_time_steps(len(plot_data[0]), 'backward')
    time_steps_y = Create_time_steps(len(plot_data[1]), 'forward')

      
    for i, x in enumerate(plot_data):
      if i: # this plots the actual future data because i is true or equal to 1 (the second item in plot data - y_train)
        plt.plot(time_steps_y, plot_data[i], marker[i], markersize=10,
                 label=labels[i])
      else: # this plots the historical data from the first item in plot data as i = 0 (x_train)
        plt.plot(time_steps_x, plot_data[i], marker[i], label=labels[i])
    plt.legend()
    plt.xlabel('Time-Step')
    return plt

def Baseline_prediction(historical_data, future_data):
    return [np.mean(historical_data)] * len(future_data)


Plot_realvpred([x_train, y_train], 'HOBBIES_1_001_CA_1_validation')


def Sum_all_columns(df):
    return df.sum()

all_sales = sales_train_validation.loc[0, 'd_1':]

for day in all_sales.index:
    all_sales[day] = sales_train_validation[day].sum()