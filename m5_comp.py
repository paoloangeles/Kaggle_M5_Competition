#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 19:51:01 2020

@author: paoloangeles
"""

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

calendar = pd.read_csv("calendar.csv")
sell_prices = pd.read_csv("sell_prices.csv")
