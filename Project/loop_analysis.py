import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
from scipy.stats import linregress


# this function is for live data processing from the take_loop experiment data_callback.


def get_binned_data(inst_data, n_periods=1, data_per_period=1000):
    """Gets the binned data ready for live plotting based on the inst_data dictionary which should contain both
    wollastons and hp data"""
    hp_data = inst_data['hallprobe'].reset_index()
    woll_data1 = inst_data['wollaston1'].reset_index()
    woll_data2 = inst_data['wollaston2'].reset_index()
    woll_data1.drop(columns=['t'], inplace=True)
    woll_data1.rename(
        columns={'det1': 'woll1det1', 'det2': 'woll1det2'}, inplace=True)
    woll_data2.drop(columns=['t'], inplace=True)
    woll_data2.rename(
        columns={'det1': 'woll2det1', 'det2': 'woll2det2'}, inplace=True)
    # concat the data into one dataframe
    data_full = pd.concat(
        [hp_data, woll_data1, woll_data2], axis=1, sort=False)

    # set the timing of each period such that it starts at 0
    # there should be an integer number of periods samples in data
    samples_per_period = int(np.floor(data_full.shape[0] / n_periods))
    for i in range(n_periods - 1):
        data_full.loc[samples_per_period * i:samples_per_period *
                                             (i + 1) - 1, "t"] -= data_full.loc[samples_per_period * i, "t"]
    # this is just not to get messed up by one extra data point which can sometimes happen
    i = n_periods - 1
    data_full.loc[samples_per_period * i:,
    "t"] -= data_full.loc[samples_per_period * i, "t"]

    t = np.array(data_full.loc[:, "t"])
    bins = np.linspace(np.min(t), np.max(t), 1000)
    data_full['bins'] = pd.cut(t, bins=bins, right=False)

    # bin in time and stack
    data_binned = data_full.groupby('bins').mean().reset_index()
    data_binned.reset_index(inplace=True)
    data_binned.drop(columns=['bins'], inplace=True)

    data_binned['diff1'] = (data_binned['woll1det1'] - data_binned['woll1det2']) / 2
    data_binned['sum1'] = (data_binned['woll1det1'] + data_binned['woll1det2']) / 2
    data_binned['diff2'] = (data_binned['woll2det1'] - data_binned['woll2det2']) / 2
    data_binned['sum2'] = (data_binned['woll2det1'] + data_binned['woll2det2']) / 2

    return data_binned

def get_raw_data(file_name):
    with h5py.File(file_name, 'r') as file:
        loops = np.array(list(file.get('/loops').keys()))
#         loops = [loops[3]]
        try:
            loop_nums = np.sort([int(l.split('data')[1]) for l in loops])
        except IndexError:
            print('No loops found in the file')
            return
        # stack all the loops
        woll_list1 = []
        for ln in loop_nums[1:]:
            wl1 = file.get('/loops/data' + str(ln) + '/wollaston1/data')[:]
            wl1[:, 0] -= wl1[0, 0]
            woll_list1.append(wl1)
        woll_list2 = []
        for ln in loop_nums[1:]:
            wl2 = file.get('/loops/data' + str(ln) + '/wollaston2/data')[:, 1:]
            woll_list2.append(wl2)
        hp_list = []
        for ln in loop_nums[1:]:
            hp = file.get('/loops/data' + str(ln) + '/hallprobe/data')[:, 1:]
            hp_list.append(hp)
    # stack the lists
    woll_full1 = np.vstack(woll_list1)
    woll_full2 = np.vstack(woll_list2)
    hp_full = np.vstack(hp_list)

    # combine the data
    data_full = np.hstack([woll_full1, woll_full2, hp_full])
    t = np.array(data_full[:, 0])
    data_fullpd = pd.DataFrame(data=data_full, columns=[
        "t", "woll1_det1", "woll1_det2", "woll2_det1", "woll2_det2", "fields_X", "fields_Y", "fields_Z"])
    return data_fullpd

def get_raw_data_one(file_name):
    with h5py.File(file_name, 'r') as file:
        wl1 = file.get('/loops/data1/wollaston1/data')[:]
        wl2 = file.get('/loops/data1/wollaston2/data')[:, 1:]
        hp = file.get('/loops/data1/hallprobe/data')[:, 1:]
        woll_full1 = wl1
        woll_full2 = wl2
        hp_full = hp

    # combine the data
    data_full = np.hstack([woll_full1, woll_full2, hp_full])
    t = np.array(data_full[:, 0])
    data_fullpd = pd.DataFrame(data=data_full, columns=[
        "t", "woll1_det1", "woll1_det2", "woll2_det1", "woll2_det2", "fields_X", "fields_Y", "fields_Z"])
    return data_fullpd

def get_averaged_loop(file_name,binnumber):
    with h5py.File(file_name, 'r') as file:
        loops = np.array(list(file.get('/loops').keys()))
#         loops = [loops[3]]
        try:
            loop_nums = np.sort([int(l.split('data')[1]) for l in loops])
        except IndexError:
            print('No loops found in the file')
            return
        # stack all the loops
        woll_list1 = []
        for ln in loop_nums[1:]:
            wl1 = file.get('/loops/data' + str(ln) + '/wollaston1/data')[:]
            wl1[:, 0] -= wl1[0, 0]
            woll_list1.append(wl1)
        woll_list2 = []
        for ln in loop_nums[1:]:
            wl2 = file.get('/loops/data' + str(ln) + '/wollaston2/data')[:, 1:]
            woll_list2.append(wl2)
        hp_list = []
        for ln in loop_nums[1:]:
            hp = file.get('/loops/data' + str(ln) + '/hallprobe/data')[:, 1:]
            hp_list.append(hp)
    # stack the lists
    woll_full1 = np.vstack(woll_list1)
    woll_full2 = np.vstack(woll_list2)
    hp_full = np.vstack(hp_list)

    # combine the data
    data_full = np.hstack([woll_full1, woll_full2, hp_full])
    t = np.array(data_full[:, 0])
    data_fullpd = pd.DataFrame(data=data_full, columns=[
        "t", "woll1_det1", "woll1_det2", "woll2_det1", "woll2_det2", "fields_X", "fields_Y", "fields_Z"])
    bins = np.linspace(np.min(t), np.max(t), binnumber)
    data_fullpd['bins'] = pd.cut(data_fullpd['t'], bins=bins, right=False)

    # bin in magnetic fields and stack
    data_binned = data_fullpd.groupby('bins').mean().reset_index()
    data_binned.dropna(inplace=True)

    data_binned['woll1_diff'] = (data_binned['woll1_det1'] - data_binned['woll1_det2']) / 2
    data_binned['woll1_sum'] = (data_binned['woll1_det1'] + data_binned['woll1_det2']) / 2
    data_binned['woll2_diff'] = (data_binned['woll2_det1'] - data_binned['woll2_det2']) / 2
    data_binned['woll2_sum'] = (data_binned['woll2_det1'] + data_binned['woll2_det2']) / 2
    # print(data_binned)
    return data_binned

def straighten_loop(data,cutoff1,cutoff2,direction,signal_type):
    if direction=='x':
        y = data[signal_type]
        bxt = np.array(data['fields_X'])
        bzt = np.array(data['fields_Z'])
        Bxsample = []
        for i in range(len(bxt)):
            aux = math.sqrt(bxt[i]**2 + bzt[i]**2)
            if bxt[i] < 0:
                Bxsample.append(-aux)
            else:
                Bxsample.append(aux)

        d = {'Bxsample': Bxsample, 'y_signal': y}
        df = pd.DataFrame(data=d)
        df1 = df[df['Bxsample'] < cutoff1]
        df2 = df[df['Bxsample'] > cutoff2]
        df1x2 = np.array(df1['Bxsample'])
        df1y2 = np.array(df1['y_signal'])
        df2x2 = np.array(df2['Bxsample'])
        df2y2 = np.array(df2['y_signal'])
        a1 = linregress(df1x2, df1y2).slope
        a2 = linregress(df2x2, df2y2).slope
        a = (a1+a2)/2
        correction = []
        for i in Bxsample:
            correction.append(i*a)
    if direction=='y':
        y = data[signal_type]
        byt = np.array(data['fields_Y'])
        d = {'Bysample': byt, 'y_signal': y}
        df = pd.DataFrame(data=d)
        df1 = df[df['Bysample'] < cutoff1]
        df2 = df[df['Bysample'] > cutoff2]
        df1x2 = np.array(df1['Bysample'])
        df1y2 = np.array(df1['y_signal'])
        df2x2 = np.array(df2['Bysample'])
        df2y2 = np.array(df2['y_signal'])
        a1 = linregress(df1x2, df1y2).slope
        a2 = linregress(df2x2, df2y2).slope
        a = (a1+a2)/2
        correction = []
        for i in byt:
            correction.append(i*a)
    if direction=='z':
        y = data[signal_type]
        bxt = np.array(data['fields_X'])
        bzt = np.array(data['fields_Z'])
        Bxsample = []
        for i in range(len(bxt)):
            aux = math.sqrt(bxt[i]**2 + bzt[i]**2)
            if bzt[i] < 0:
                Bxsample.append(-aux)
            else:
                Bxsample.append(aux)

        d = {'Bxsample': Bxsample, 'y_signal': y}
        df = pd.DataFrame(data=d)
        df1 = df[df['Bxsample'] < cutoff1]
        df2 = df[df['Bxsample'] > cutoff2]
        df1x2 = np.array(df1['Bxsample'])
        df1y2 = np.array(df1['y_signal'])
        df2x2 = np.array(df2['Bxsample'])
        df2y2 = np.array(df2['y_signal'])
        a1 = linregress(df1x2, df1y2).slope
        a2 = linregress(df2x2, df2y2).slope
        a = (a1+a2)/2
        correction = []
        for i in Bxsample:
            correction.append(i*a)
    

    newy = np.array(y - correction)
    mn = np.min(newy)
    mx = np.max(newy)
    dif = abs(mx - mn)
    if mn < 0:
        newy = newy + abs(mn) - dif/2
    else:
        newy = newy - abs(mn) - dif/2
    
    
    return newy, a

def plot_single_loop(filepath, direction, loop_number):    
    with h5py.File(filepath, 'r') as file:    
        hall_data = np.array(list(file.get('/loops/data' + str(loop_number) + '/hallprobe/data')))
        woll1_data = np.array(list(file.get('/loops/data' + str(loop_number) + '/wollaston1/data')))
        woll2_data = np.array(list(file.get('/loops/data' + str(loop_number) + '/wollaston2/data')))
        
    w1d1 = woll1_data[:,1]
    w1d2 = woll1_data[:,2]
    w2d1 = woll2_data[:,1]
    w2d2 = woll2_data[:,2]
    hall = hall_data[:,direction]
    w1dif = w1d1-w1d2
    w2dif = w2d1-w2d2
    return w1dif,w2dif,hall

def get_magnetic_field(filepath, direction, loop_number):
    with h5py.File(filepath, 'r') as file:    
        hall_data = np.array(list(file.get('/loops/data' + str(loop_number) + '/hallprobe/data')))
    hall = hall_data[:,direction]
    return hall

def plot_loop(data, field_direction='X'):
    field_direction = 'fields_' + field_direction
    fig = plt.figure(figsize=(6, 6))
    plt.rcParams.update({'font.size': 22})

    mn1 = np.min(data['woll1_diff'])
    mx1 = np.max(data['woll1_diff'])
    plt.plot(data[field_direction],
             (data['woll1_diff'] - mn1) / (mx1 - mn1), label='wollaston 1')
    mn2 = np.min(data['woll2_diff'])
    mx2 = np.max(data['woll2_diff'])
    # plt.plot(data[field_direction],
    #          (data['woll2_diff'] - mn2) / (mx2 - mn2), label='wollaston 2')

    plt.xlabel('B [mT]')
    plt.ylabel('MOKE signal')
    plt.grid()
    plt.tight_layout()


if __name__ == '__main__':
    folder_name = r'C:\Users\user\Documents\3DMOKE'
    file_name = 'LoopTaking_20191028-164515.h5'
    field_direction = 'X'


    path = os.path.join(folder_name, file_name)
    data = get_averaged_loop(path)
    plot_loop(data, field_direction=field_direction)
    # plt.savefig('First.png')
    plt.show()
    

