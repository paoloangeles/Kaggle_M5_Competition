# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 12:07:30 2020

@author: Paolo
"""


## Need to iterate over all products and all stores 
HOBBIES_1_001_CA_1 = sell_prices.loc[(sell_prices['store_id'] == 'CA_1') & (sell_prices['item_id'] == 'HOBBIES_1_001')]


## Create feature matrix
sell_price_features = pd.DataFrame()

## Create appropriate reference indexes
sell_price_features['d'] = calendar['d']
sell_price_features['wm_yr_wk'] = calendar['wm_yr_wk']

## Merge sell prices to feature matrix
sell_price_features = sell_price_features.merge(HOBBIES_1_001_CA_1, on='wm_yr_wk', how='left')

## Drop all nans
sell_price_features = sell_price_features.drop(sell_price_features[sell_price_features['sell_price'].astype(str) == 'nan'].index)

## Create feature columns
sell_price_features['week_month'] = ""
sell_price_features['week_halfyear'] = ""
sell_price_features['week_year'] = ""
sell_price_features['week_all'] = ""


## Define specific time horizons
start_wm_yr_wk = sell_price_features.loc[sell_price_features.index[0], 'wm_yr_wk']
weeks_in_month = 4
weeks_in_halfyear = 26
weeks_in_year = 52


## Iterate through all weeks with sell prices
for week in sell_price_features['wm_yr_wk'].unique():
    
    ## Assign appropriate indexes for each time horizon (if start horizon index is less than original start index, this is not possible so force value to original start index)
    if week - weeks_in_month < start_wm_yr_wk:
        month_index = start_wm_yr_wk
    else:
        month_index = week - weeks_in_month
        
    if week - weeks_in_halfyear < start_wm_yr_wk:
        halfyear_index = start_wm_yr_wk
    else:
        halfyear_index = week - weeks_in_halfyear
        
    if week - weeks_in_year < start_wm_yr_wk:
        year_index = start_wm_yr_wk
    else:
        year_index = week - weeks_in_year
    
    av_week = sell_price_features.loc[sell_price_features['wm_yr_wk'] == week, 'sell_price'].mean()
    av_month = sell_price_features.loc[(sell_price_features['wm_yr_wk'] >= month_index) & (sell_price_features['wm_yr_wk'] <= week), 'sell_price'].mean()
    av_halfyear = sell_price_features.loc[(sell_price_features['wm_yr_wk'] >= halfyear_index) & (sell_price_features['wm_yr_wk'] <= week), 'sell_price'].mean()
    av_year = sell_price_features.loc[(sell_price_features['wm_yr_wk'] >= year_index) & (sell_price_features['wm_yr_wk'] <= week), 'sell_price'].mean()
    av_all = sell_price_features.loc[(sell_price_features['wm_yr_wk'] >= start_wm_yr_wk) & (sell_price_features['wm_yr_wk'] <= week), 'sell_price'].mean()
    
    time_horizon = [av_month, av_halfyear, av_year, av_all]
    column_names = ['week_month', 'week_halfyear', 'week_year', 'week_all']
    
    ## iterate for each time horizon
    for i, val in enumerate(time_horizon):
        ## if weekly average is 0-25% lower, assign a 1
        if (((av_week - val)/val) > -0.025) and (((av_week - val)/val) < 0):
            sell_price_features.loc[sell_price_features['wm_yr_wk'] == week, column_names[i]] = 1
        ## if weekly average is 25-50% lower, assign a 2
        elif (((av_week - val)/val) <= -0.025) and (((av_week - val)/val) > -0.05):
            sell_price_features.loc[sell_price_features['wm_yr_wk'] == week, column_names[i]] = 2
        ## if weekly average is 50%+ lower, assign a 3
        elif ((av_week - val)/val) < -0.05:
            sell_price_features.loc[sell_price_features['wm_yr_wk'] == week, column_names[i]] = 3
        ## if weekly average is 0-25% higher, assign a 4
        if (((av_week - val)/val) < 0.025) and (((av_week - val)/val) >= 0):
            sell_price_features.loc[sell_price_features['wm_yr_wk'] == week, column_names[i]] = 4
        ## if weekly average is 25-50% higher, assign a 5
        elif (((av_week - val)/val) >= 0.025) and (((av_week - val)/val) < 0.05):
            sell_price_features.loc[sell_price_features['wm_yr_wk'] == week, column_names[i]] = 5
        ## if weekly average is 50%+ higher, assign a 6
        elif ((av_week - val)/val) > 0.05:
            sell_price_features.loc[sell_price_features['wm_yr_wk'] == week, column_names[i]] = 6