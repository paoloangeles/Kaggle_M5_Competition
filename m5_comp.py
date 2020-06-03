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

def Plot_item(product_id):

####### PLOT ALL DATA #######
    
    
    # select item, reindex to id and keep sales data columns, transform, plot
    
    sales_train_validation.loc[sales_train_validation['id'] == product_id, sales_columns].T.plot(figsize=(15, 5), title = 'HOBBIES_1_001_CA_1 sales by "d" number', color = next(color_cycle))
    plt.legend('')
    plt.show()
    
    example = sales_train_validation.loc[sales_train_validation['id'] == product_id, sales_columns]
    
    # def Plot_range(product_id, start_index_sales, start_point):

Plot_item('HOBBIES_1_001_CA_1_validation')

def Plot_item_range(product_id, start_day, day_range, num_previous_ranges):

    start_index = []
    start_index[0] = 'd_' + str(start_day)
    end_index = 'd_' + str(start_day + day_range)
    
    start_indexes = list(range(start_day, day_range+1))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for start_point in range(num_previous_ranges, start_day):
    
        start_index = ((start_point-1)*days)+start_index_sales
        end_index = (start_point*days)+start_index_sales+1
        
        x = list(range(start_index, end_index+1))
        y = sales_train_validation.loc[sales_train_validation['id'] == product_id, "d_" + str(start_index):"d_" + str(end_index)].T
    
    plt.plot(x_start, y)
    plt.xlabel('Day number')
    plt.ylabel('Number of sales of ' + str(product_id))

####### Check for event between specified days #######

# check if event is present between days

# calendar_dates = calendar[(calender.d >= 'd_' + str(start_index)) & (calendar.d <= 'd_' + str(end_index)), 7:10]

## extract calendar with only event days and then extract day_id from that new calendar

test = []
for i in range(len(x_start)):
    test.append('d_' + str(x_start[i]))

calendar_events = calendar[['d', 'date', 'event_name_1', 'event_name_2', 'event_type_1', 'event_type_2', 'snap_CA']] ## need double brackets to give a list, do not need double brackets if single column

calendar_events[calendar_events['d'].isin(test)].date