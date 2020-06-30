# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 09:52:44 2020

@author: Paolo
"""
import numpy as np
import pandas as pd
# import plotly.express as px
import gc
import joblib
from lightgbm import LGBMRegressor

# DATA IMPORT ##
# Import all tables
sales = pd.read_csv('sales_train_evaluation.csv')
sales.name = 'sales'
calendar = pd.read_csv('calendar.csv')
calendar.name = 'calendar'
prices = pd.read_csv('sell_prices.csv')
prices.name = 'prices'


## DATA PREPARATION ##
## Add zero sales for the remaining days 1942-1969 because we have the validation set for d_1914 to d_1941
for d in range(1942,1970):
    col = 'd_' + str(d)
    sales[col] = 0
    sales[col] = sales[col].astype(np.int16)

## DOWNCAST ##
## Downcast dataframes to reduce storage and expedite operation process
## Find maximum values of each numerical column and downcast to appropriate type
## For categorical columns, pandas stores them as objects. Map unique categories with integers (encoding).

def Downcast(df):
    cols = df.dtypes.index.tolist() ## put column names into list
    types = df.dtypes.values.tolist() ## put column types into list
    
    for i,t in enumerate(types):
        if 'int' in str(t):
            if df[cols[i]].min() > np.iinfo(np.int8).min and df[cols[i]].max() < np.iinfo(np.int8).max:
                df[cols[i]] = df[cols[i]].astype(np.int8)
            elif df[cols[i]].min() > np.iinfo(np.int16).min and df[cols[i]].max() < np.iinfo(np.int16).max:
                df[cols[i]] = df[cols[i]].astype(np.int16)
            elif df[cols[i]].min() > np.iinfo(np.int32).min and df[cols[i]].max() < np.iinfo(np.int32).max:
                df[cols[i]] = df[cols[i]].astype(np.int32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.int64)
        elif 'float' in str(t):
            if df[cols[i]].min() > np.finfo(np.float16).min and df[cols[i]].max() < np.finfo(np.float16).max:
                df[cols[i]] = df[cols[i]].astype(np.float16)
            elif df[cols[i]].min() > np.finfo(np.float32).min and df[cols[i]].max() < np.finfo(np.float32).max:
                df[cols[i]] = df[cols[i]].astype(np.float32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.float64)
        elif t == np.object:
            if cols[i] == 'date':
                df[cols[i]] = pd.to_datetime(df[cols[i]], format='%Y-%m-%d')
            else:
                df[cols[i]] = df[cols[i]].astype('category')
    return df


sales = Downcast(sales)
prices = Downcast(prices)
calendar = Downcast(calendar)

sales.to_pickle("./sales.pkl")
calendar.to_pickle("./calendar.pkl")
prices.to_pickle("./prices.pkl")

sales = pd.read_pickle("./sales.pkl")
calendar = pd.read_pickle("./calendar.pkl")
prices = pd.read_pickle("./prices.pkl")

## MELT & COMBINE ALL DATAFRAMES ##
## Melt Sales dataframe to have key column identifiers of ID, item ID, department ID, category ID, store ID and state ID
## Melt unpivots columns not specified by id_vars to be used as values in the table
## id_vars are the columns which will not be unpivoted - will still be used as the reference index


df = pd.melt(sales, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name='d', value_name='sold').dropna()

## Merge calendar and prices dataframes into df
df = pd.merge(df, calendar, on='d', how='left')
df = pd.merge(df, prices, on=['store_id','item_id','wm_yr_wk'], how='left') 


# ## EXPLORATORY DATA ANALYSIS ## 
# ## Check average sale price for each item
# ## the groupby function, groups together specific columns and column values. Here I specify that I want the
# ## average sell price to be from each state, store and item
# group_price_store = df.groupby(['state_id','store_id','item_id'],as_index=False)['sell_price'].mean().dropna()

# ## Using plotly express, plot the prices vs the store. Display different colours for the state and use the hover
# ## property to identify which item is present
# fig = px.violin(group_price_store, x='store_id', color='state_id', y='sell_price',box=True, hover_name='item_id')
# fig.update_xaxes(title_text='Store')
# fig.update_yaxes(title_text='Selling Price($)')
# fig.update_layout(template='seaborn',title='Distribution of Items prices wrt Stores',legend_title_text='State')
# fig.show()


## FEATURE ENGINEERING ## 
## Label encoding
## Store the different categories found in each column with their corresponding codes
d_id = dict(zip(df.id.cat.codes, df.id))
d_item_id = dict(zip(df.item_id.cat.codes, df.item_id))
d_dept_id = dict(zip(df.dept_id.cat.codes, df.dept_id))
d_cat_id = dict(zip(df.cat_id.cat.codes, df.cat_id))
d_store_id = dict(zip(df.store_id.cat.codes, df.store_id))
d_state_id = dict(zip(df.state_id.cat.codes, df.state_id))

## Label encode categorical features
df.d = df['d'].apply(lambda x: x.split('_')[1]).astype(np.int16) ## change day number to regular integer number
## Convert categories to code - label encode
cols = df.dtypes.index.tolist()
types = df.dtypes.values.tolist()
for i,type in enumerate(types):
    if type.name == 'category':
        df[cols[i]] = df[cols[i]].cat.codes
        
## Remove date as features present
df.drop('date',axis=1,inplace=True)    

## Create lag features
lags = [1,2,3,6,12,24,36]
for lag in lags:
    df['sold_lag_'+str(lag)] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],as_index=False)['sold'].shift(lag).astype(np.float16)
    
    
## Create mean features grouped by particular categories
df['iteam_sold_avg'] = df.groupby('item_id')['sold'].transform('mean').astype(np.float16)
df['state_sold_avg'] = df.groupby('state_id')['sold'].transform('mean').astype(np.float16)
df['store_sold_avg'] = df.groupby('store_id')['sold'].transform('mean').astype(np.float16)
df['cat_sold_avg'] = df.groupby('cat_id')['sold'].transform('mean').astype(np.float16)
df['dept_sold_avg'] = df.groupby('dept_id')['sold'].transform('mean').astype(np.float16)
df['cat_dept_sold_avg'] = df.groupby(['cat_id','dept_id'])['sold'].transform('mean').astype(np.float16)
df['store_item_sold_avg'] = df.groupby(['store_id','item_id'])['sold'].transform('mean').astype(np.float16)
df['cat_item_sold_avg'] = df.groupby(['cat_id','item_id'])['sold'].transform('mean').astype(np.float16)
df['dept_item_sold_avg'] = df.groupby(['dept_id','item_id'])['sold'].transform('mean').astype(np.float16)
df['state_store_sold_avg'] = df.groupby(['state_id','store_id'])['sold'].transform('mean').astype(np.float16)
df['state_store_cat_sold_avg'] = df.groupby(['state_id','store_id','cat_id'])['sold'].transform('mean').astype(np.float16)
df['store_cat_dept_sold_avg'] = df.groupby(['store_id','cat_id','dept_id'])['sold'].transform('mean').astype(np.float16)
    
## Calculate rolling window (window of 7 days) mean for items sold
df['rolling_sold_mean'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])['sold'].transform(lambda x: x.rolling(window=7).mean()).astype(np.float16)

## Calculte expanding window (minimum window period of 2 days) mean for items sold
df['expanding_sold_mean'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])['sold'].transform(lambda x: x.expanding(2).mean()).astype(np.float16)

## Calculate trend if number sold is higher or lower than all time (positive vs negative)
df['daily_avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id','d'])['sold'].transform('mean').astype(np.float16)
df['avg_sold'] = df.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])['sold'].transform('mean').astype(np.float16)
df['selling_trend'] = (df['daily_avg_sold'] - df['avg_sold']).astype(np.float16)
df.drop(['daily_avg_sold','avg_sold'],axis=1,inplace=True)

## Because of the 36 day lag feature, there will be features as NaN for the first 35 days
df = df[df['d']>=36]


## MODELLING AND PREDICTION ##
## Split data up to to use for validation and testing
data = df
## validation set = day 1914 to 1942
valid = data[(data['d']>=1914) & (data['d']<1942)][['id','d','sold']]
## testing set = day 1942 onwards
test = data[data['d']>=1942][['id','d','sold']]

## assign numbers sold to prediction
eval_preds = test['sold']
valid_preds = valid['sold']

## Conduct model training based on individual stores
## Get unique store ids
sales = pd.read_pickle('./sales.pkl')
stores = sales.store_id.cat.codes.unique().tolist()

## Run model
for store in stores:
    df = data[data['store_id']==store]
    
    #Split the data
    X_train, y_train = df[df['d']<1914].drop('sold',axis=1), df[df['d']<1914]['sold']
    X_valid, y_valid = df[(df['d']>=1914) & (df['d']<1942)].drop('sold',axis=1), df[(df['d']>=1914) & (df['d']<1942)]['sold']
    X_test = df[df['d']>=1942].drop('sold',axis=1)
    
    #Train and validate
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.3,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=8,
        num_leaves=50,
        min_child_weight=300
    )
    print('*****Prediction for Store: {}*****'.format(d_store_id[store]))
    model.fit(X_train, y_train, eval_set=[(X_train,y_train),(X_valid,y_valid)],
             eval_metric='rmse', verbose=20, early_stopping_rounds=20)
    valid_preds[X_valid.index] = model.predict(X_valid)
    eval_preds[X_test.index] = model.predict(X_test)
    filename = 'model'+str(d_store_id[store])+'.pkl'
    # save model
    joblib.dump(model, filename)
    del model, X_train, y_train, X_valid, y_valid
    gc.collect()