# -*- coding: utf-8 -*-
"""
Created on Tue May 19 08:08:24 2020

@author: Paolo
"""

# Import modules

import pandas as pd

# Get data from csvs
calendar = pd.read_csv("calendar.csv")
sell_prices = pd.read_csv("sell_prices.csv")
sales_train_validation = pd.read_csv("sales_train_validation.csv")
sales_train_eval = pd.read_csv("sales_train_evaluation.csv")

calendar.to_pickle("./calendar.pkl")
sell_prices.to_pickle("./sell_prices.pkl")
sales_train_validation.to_pickle("./sales_train_validation.pkl")
sales_train_eval.to_pickle("./sales_train_evaluation.pkl")