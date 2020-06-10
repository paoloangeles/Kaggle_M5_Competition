# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:40:26 2020

@author: Paolo
"""


def num_days_after_event(data):
    dataOut = np.empty(len(data)) ## Create empty array same length as data input
    dataOut[:] = np.NaN ## Fill all elements with NaN values
    eventOccur = np.array([i for i, val in enumerate(data) if val]) ## list indexes (days) for which an event occurs (is true)
    eventOccur = eventOccur + 1 ## add one to the index
    # eventOccur = np.insert(eventOccur, 0, 0) ## insert a value of 0 at the start of the array
    for event, nextEvent in zip(eventOccur[:-1], eventOccur[1:]): ## Creates countdown array to each event occurence
        dataOut[event:nextEvent] = list(range(1, nextEvent-event+1))
    return dataOut

## Days leading to or after any event occurence
any_event_occur = np.logical_or(calendar['event_name_1'] != 'nan', calendar['event_name_2'] != 'nan')
features['days_until_event'] = num_days_until_event(any_event_occur)
features['days_after_event'] = num_days_after_event(any_event_occur)