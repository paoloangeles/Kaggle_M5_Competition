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

# Plot visualisation

# Product id
product_id = 0

# Time constraints
start_index_sales = 6
year = 25
days_in_year = 365

sales_train_validation.iloc[product_id, ((year-1)*days_in_year)+start_index_sales:(year*days_in_year)+start_index_sales+1].plot.line()