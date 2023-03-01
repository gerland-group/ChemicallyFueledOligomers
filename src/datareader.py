#!/bin/env python3

import pandas as pd
import numpy as np



def read_experimental_data_T25_EDC10(dir=''):
    #df = pd.read_excel("/space/origin3/ChemicallyFueledOligomers_Solution_RateKernel_Quadratic/src/data_experiment/data_T25_EDC10.xlsx")
    if(dir!=''):
        df = pd.read_excel(dir)
    else:
        df = pd.read_excel("../data_experiment/data_T25_EDC10.xlsx")

    # CAVEAT: need to check carefully if the keys are correct for each data sheet!
    times = np.asarray(df['time (hr)']) # in units h
    c_EDC = np.asarray(df['EDC (mM)']) # in units mM
    c_EDC_err = np.asarray(df['error']) # in units mM
    c_T2 = np.asarray(df['T2 (mM)']) # in units mM
    c_T2_err = np.asarray(df['error.1']) # in units mM
    c_T3 = np.asarray(df['T3 (mM)']) # in units mM
    c_T3_err = np.asarray(df['error.2']) # in units mM
    c_T4 = np.asarray(df['T4 (mM)']) # in units mM
    c_T4_err = np.asarray(df['error.3']) # in units mM
    c_T5 = np.asarray(df['T5 (mM)']) # in units mM
    c_T5_err = np.asarray(df['error.4']) # in units mM
    c_T1loss = np.asarray(df['T1* (mM)']) # in units mM
    c_T1loss_err = np.asarray(df['error.6']) # in units mM

    map_cs = {'2,_,0' : 0, '3,_,0' : 1, '4,_,0' : 2, '5,_,0' : 3, '1,_,2' : 4}
    cs = np.array([c_T2, c_T3, c_T4, c_T5, c_T1loss])
    cs_err = np.array([c_T2_err, c_T3_err, c_T4_err, c_T5_err, c_T1loss_err])

    return times, map_cs, cs, cs_err, c_EDC, c_EDC_err



def read_experimental_data_T25_EDC10_A10(dir=''):
    #df = pd.read_excel("./data_experiment/data_T25_EDC10_A100.32.xlsx")
    if(dir!=''):
        df = pd.read_excel(dir)
    else:
        df = pd.read_excel("../data_experiment/data_T25_EDC10_A10-08.xlsx")

    # CAVEAT: need to check carefully if the keys are correct for each data sheet!
    times = np.asarray(df['time (hr)']) # in units h
    c_EDC = np.asarray(df['EDC (mM)']) # in units mM
    c_EDC_err = np.asarray(df['error']) # in units mM
    c_T2 = np.asarray(df['T2 (mM)']) # in units mM
    c_T2_err = np.asarray(df['error.1']) # in units mM
    c_T3 = np.asarray(df['T3 (mM)']) # in units mM
    c_T3_err = np.asarray(df['error.2']) # in units mM
    c_T4 = np.asarray(df['T4 (mM)']) # in units mM
    c_T4_err = np.asarray(df['error.3']) # in units mM
    c_T5 = np.asarray(df['T5 (mM)']) # in units mM
    c_T5_err = np.asarray(df['error.4']) # in units mM
    c_T1loss = np.asarray(df['T1* (mM)']) # in units mM
    c_T1loss_err = np.asarray(df['error.6']) # in units mM
    c_T3loss = np.asarray(df['T3* (mM)']) # in units mM
    c_T3loss_err = np.asarray(df['error.5']) # in units mM

    maps_cs = {'2,_,0' : 0, '3,_,0' : 1, '4,_,0' : 2, '5,_,0' : 3, '1,_,2' : 4, '3,_,2' : 5}
    cs = np.array([c_T2, c_T3, c_T4, c_T5, c_T1loss, c_T3loss])
    cs_err = np.array([c_T2_err, c_T3_err, c_T4_err, c_T5_err, c_T1loss_err, c_T3loss_err])

    return times, maps_cs, cs, cs_err, c_EDC, c_EDC_err



def read_parameters_system_with_template(filepath, name, Lt):

    # read existing excel-file
    df = pd.read_excel(filepath)

    # identify the index that is associated with the name of the data-set of interest
    index = np.where(np.asarray(df['name'])==name)[0]
    if(len(index) != 1):
        raise ValueError("name not existing or appearing multiple times")
    index = index[0] # get rid of array structure, take only the entry

    # identify the reaction rate constants on the template
    klig_temp = np.asarray(df['klig_temp'])[index]
    kloss_temp = np.asarray(df['kloss_temp'])[index]
    khydro_temp = np.asarray(df['khydro_temp'])[index]
    kcut_temp = np.asarray(df['kcut_temp'])[index]
    ks_temp = np.array([klig_temp, kloss_temp, khydro_temp, kcut_temp])

    # identify the binding affinities
    Kd1 = np.asarray(df['Kd1'])[index]
    Kd2 = np.asarray(df['Kd2'])[index]
    Kd3 = np.asarray(df['Kd3'])[index]
    Kd4 = np.asarray(df['Kd4'])[index]
    KdRest = np.asarray(df['KdRest'])[index]
    KdRests = [KdRest for i in range(Lt-4)]
    Kds = np.array([Kd1, Kd2, Kd3, Kd4, *KdRests])

    # create output array
    ks_temp_Kds_log = np.concatenate([np.log2(ks_temp), np.log2(Kds)])

    return ks_temp_Kds_log
