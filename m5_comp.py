#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 19:51:01 2020

@author: paoloangeles
"""

# Import modules

import pandas as pd

# Get data from pickle files
calendar = pd.read_pickle("./calendar.pkl")
sell_prices = pd.read_pickle("./sell_prices.pkl")
sales_train_validation = pd.read_pickle("./sales_train_validation.pkl")

## Data preprocessing
## Check for any NaN values
total_num_days = 6
for day in range(total_num_days):
    null_rows = sales_train_validation["d_" + str(day+1)].isnull().any()
    print(null_rows)

# Plot visualisation

# Product id
product_id = 0

# Time constraints
start_index_sales = 6 # initial columns are not sales numbers
start_point = 50

period_length = "month"
if period_length == "month":
    days = 28
elif period_length == "year":
    days = 365

sales_train_validation.iloc[product_id, ((start_point-1)*days)+start_index_sales:(start_point*days)+start_index_sales+1].plot.line()