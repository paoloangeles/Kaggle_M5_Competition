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
    dataOut = np.empty(len(data)) ## Create empty array same length as data input
    dataOut[:] = np.NaN ## Fill all elements with NaN values
    eventOccur = np.array([i for i, val in enumerate(data) if val]) ## list indexes (days) for which an event occurs (is true)
    eventOccur = eventOccur + 1 ## add one to the index
    eventOccur = np.insert(eventOccur, 0, 0) ## insert a value of 0 at the start of the array
    for event, nextEvent in zip(eventOccur[:-1], eventOccur[1:]): ## Creates countdown array to each event occurence
        dataOut[event:nextEvent] = list(reversed(range(nextEvent-event)))
    return dataOut

def num_days_after_event(data):
    dataOut = np.empty(len(data)) ## Create empty array same length as data input
    dataOut[:] = np.NaN ## Fill all elements with NaN values
    eventOccur = np.array([i for i, val in enumerate(data) if val]) ## list indexes (days) for which an event occurs (is true)
    eventOccur = eventOccur + 1 ## add one to the index
    # eventOccur = np.insert(eventOccur, 0, 0) ## insert a value of 0 at the start of the array
    for event, nextEvent in zip(eventOccur[:-1], eventOccur[1:]): ## Creates countdown array to each event occurence
        dataOut[event:nextEvent] = list(range(1, nextEvent-event+1))
    return dataOut

gaus = signal.gaussian(365, 100)
features = pd.DataFrame()

# Keep this column as key for any future merges/ reference
features['d'] = example['d']

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

## Days leading to or after any event occurence
calendar['event_name_1'] = calendar['event_name_1'].astype(str)
calendar['event_name_2'] = calendar['event_name_2'].astype(str)
event_occur = np.logical_or(calendar['event_name_1'] != 'nan', calendar['event_name_2'] != 'nan')
features['days_until_event'] = num_days_until_event(event_occur[:len(sales_columns)]) ## limit to sales_columns as these are the only days we have data for
features['days_after_event'] = num_days_after_event(event_occur[:len(sales_columns)]) ## limit to sales_columns as these are the only days we have data for

# SNAP - US gov provides benefits for low income families and individuals to purchase food products
## extract occurences into feature dataframe
features['SNAP_CA'] = calendar['snap_CA']
features['SNAP_TX'] = calendar['snap_TX']
features['SNAP_WI'] = calendar['snap_WI']


## Begin sell prices feature creation
# Bring common index to feature dataframe - wm_yr_wk
features['wm_yr_wk'] = calendar['wm_yr_wk']


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
    
    
    
## Create feature matrix for all stores and items
sell_price_features_final = pd.DataFrame()
sell_price_features_final['wm_yr_wk'] = ""
sell_price_features_final['store_id'] = ""
sell_price_features_final['item_id'] = ""
sell_price_features_final['sell_price'] = ""
sell_price_features_final['week_month'] = ""
sell_price_features_final['week_halfyear'] = ""
sell_price_features_final['week_year'] = ""
sell_price_features_final['week_all'] = ""



## Create feature matrix
sell_price_features = pd.DataFrame()

## Create appropriate reference indexes
# sell_price_features['d'] = calendar['d']
sell_price_features['wm_yr_wk'] = calendar['wm_yr_wk'].unique()

## Merge sell prices to feature matrix
sell_price_features = sell_price_features.merge(sell_prices, on='wm_yr_wk', how='left')

## Drop all nans
sell_price_features = sell_price_features.drop(sell_price_features[sell_price_features['sell_price'].astype(str) == 'nan'].index)

sell_price_features['month_average_price'] = sell_price_features.groupby(['store_id', 'item_id'])['sell_price'].transform(lambda x: x.rolling(4, min_periods = 1).mean())
sell_price_features['half_year_average_price'] = sell_price_features.groupby(['store_id', 'item_id'])['sell_price'].transform(lambda x: x.rolling(26, min_periods = 1).mean())
sell_price_features['year_average_price'] = sell_price_features.groupby(['store_id', 'item_id'])['sell_price'].transform(lambda x: x.rolling(52, min_periods = 1).mean())
sell_price_features['average_price'] = sell_price_features.groupby(['store_id', 'item_id'])['sell_price'].transform(lambda x: x.expanding().mean())



sell_price_features = pd.read_pickle("./sell_price_features.pkl")

def assign_sell_price_feature(price):
    
    if (((price[0] - price[1])/price[1]) > -0.025) and (((price[0] - price[1])/price[1]) < 0):
        return 3
    ## if weekly average is 25-50% lower, assign a 2
    elif (((price[0] - price[1])/price[1]) <= -0.025) and (((price[0] - price[1])/price[1]) > -0.05):
        return 2
    ## if weekly average is 50%+ lower, assign a 3
    elif ((price[0] - price[1])/price[1]) < -0.05:
        return 1
    ## if weekly average is 0-25% higher, assign a 4
    if (((price[0] - price[1])/price[1]) < 0.025) and (((price[0] - price[1])/price[1]) >= 0):
        return 4
    ## if weekly average is 25-50% higher, assign a 5
    elif (((price[0] - price[1])/price[1]) >= 0.025) and (((price[0] - price[1])/price[1]) < 0.05):
        return 5
    ## if weekly average is 50%+ higher, assign a 6
    elif ((price[0] - price[1])/price[1]) > 0.05:
        return 6

sell_price_features['week_month'] = sell_price_features[['sell_price', 'month_average_price']].apply(assign_sell_price_feature, axis = 1)
sell_price_features['week_halfyear'] = sell_price_features[['sell_price', 'half_year_average_price']].apply(assign_sell_price_feature, axis = 1)
sell_price_features['week_year'] = sell_price_features[['sell_price', 'year_average_price']].apply(assign_sell_price_feature, axis = 1)
sell_price_features['week_all'] = sell_price_features[['sell_price', 'average_price']].apply(assign_sell_price_feature, axis = 1)

sell_price_features.to_pickle("./sell_price_features.pkl")

# for store in sell_prices['store_id'].unique():
#     for item in sell_prices['item_id'].unique():

#         ## Need to iterate over all products and all stores 
#         sell_price_DF = sell_prices.loc[(sell_prices['store_id'] == store) & (sell_prices['item_id'] == item)]
        
        
#         ## Create feature matrix
#         sell_price_features = pd.DataFrame()
        
#         ## Create appropriate reference indexes
#         # sell_price_features['d'] = calendar['d']
#         sell_price_features['wm_yr_wk'] = calendar['wm_yr_wk'].unique()
        
#         ## Merge sell prices to feature matrix
#         sell_price_features = sell_price_features.merge(sell_price_DF, on='wm_yr_wk', how='left')
        
#         ## Drop all nans
#         sell_price_features = sell_price_features.drop(sell_price_features[sell_price_features['sell_price'].astype(str) == 'nan'].index)
        
#         ## Create feature columns
#         sell_price_features['week_month'] =  ""
#         sell_price_features['week_halfyear'] = ""
#         sell_price_features['week_year'] = ""
#         sell_price_features['week_all'] = ""
        
        
#         ## Define specific time horizons
#         start_wm_yr_wk = sell_price_features.loc[sell_price_features.index[0], 'wm_yr_wk']
#         weeks_in_month = 4
#         weeks_in_halfyear = 26
#         weeks_in_year = 52
        
        
#         ## Iterate through all weeks with sell prices
#         for week in sell_price_features['wm_yr_wk'].unique():
            
#             ## Assign appropriate indexes for each time horizon (if start horizon index is less than original start index, this is not possible so force value to original start index)
#             if week - weeks_in_month < start_wm_yr_wk:
#                 month_index = start_wm_yr_wk
#             else:
#                 month_index = week - weeks_in_month
                
#             if week - weeks_in_halfyear < start_wm_yr_wk:
#                 halfyear_index = start_wm_yr_wk
#             else:
#                 halfyear_index = week - weeks_in_halfyear
                
#             if week - weeks_in_year < start_wm_yr_wk:
#                 year_index = start_wm_yr_wk
#             else:
#                 year_index = week - weeks_in_year
            
#             av_week = sell_price_features.loc[sell_price_features['wm_yr_wk'] == week, 'sell_price'].mean()
#             av_month = sell_price_features.loc[(sell_price_features['wm_yr_wk'] >= month_index) & (sell_price_features['wm_yr_wk'] <= week), 'sell_price'].mean()
#             av_halfyear = sell_price_features.loc[(sell_price_features['wm_yr_wk'] >= halfyear_index) & (sell_price_features['wm_yr_wk'] <= week), 'sell_price'].mean()
#             av_year = sell_price_features.loc[(sell_price_features['wm_yr_wk'] >= year_index) & (sell_price_features['wm_yr_wk'] <= week), 'sell_price'].mean()
#             av_all = sell_price_features.loc[(sell_price_features['wm_yr_wk'] >= start_wm_yr_wk) & (sell_price_features['wm_yr_wk'] <= week), 'sell_price'].mean()
            
#             time_horizon = [av_month, av_halfyear, av_year, av_all]
#             column_names = ['week_month', 'week_halfyear', 'week_year', 'week_all']
            
#             ## iterate for each time horizon
#             for i, val in enumerate(time_horizon):
#                 ## if weekly average is 0-25% lower, assign a 1
#                 if (((av_week - val)/val) > -0.025) and (((av_week - val)/val) < 0):
#                     sell_price_features.loc[sell_price_features['wm_yr_wk'] == week, column_names[i]] = 1
#                 ## if weekly average is 25-50% lower, assign a 2
#                 elif (((av_week - val)/val) <= -0.025) and (((av_week - val)/val) > -0.05):
#                     sell_price_features.loc[sell_price_features['wm_yr_wk'] == week, column_names[i]] = 2
#                 ## if weekly average is 50%+ lower, assign a 3
#                 elif ((av_week - val)/val) < -0.05:
#                     sell_price_features.loc[sell_price_features['wm_yr_wk'] == week, column_names[i]] = 3
#                 ## if weekly average is 0-25% higher, assign a 4
#                 if (((av_week - val)/val) < 0.025) and (((av_week - val)/val) >= 0):
#                     sell_price_features.loc[sell_price_features['wm_yr_wk'] == week, column_names[i]] = 4
#                 ## if weekly average is 25-50% higher, assign a 5
#                 elif (((av_week - val)/val) >= 0.025) and (((av_week - val)/val) < 0.05):
#                     sell_price_features.loc[sell_price_features['wm_yr_wk'] == week, column_names[i]] = 5
#                 ## if weekly average is 50%+ higher, assign a 6
#                 elif ((av_week - val)/val) > 0.05:
#                     sell_price_features.loc[sell_price_features['wm_yr_wk'] == week, column_names[i]] = 6
                    
#         sell_price_features_final = sell_price_features_final.append(sell_price_features, ignore_index = True)
