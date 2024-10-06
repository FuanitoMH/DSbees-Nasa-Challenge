# Import libraries
import numpy as np
import pandas as pd
import obspy
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

cat_directory = './data/lunar/training/catalogs/'
cat_file = cat_directory + 'apollo12_catalog_GradeA_final.csv'
cat = pd.read_csv(cat_file)

ls_filenames = cat['filename'].tolist()
ls_arrival_times = cat['time_rel(sec)'].tolist()

for filename, arrival_time in zip(ls_filenames, ls_arrival_times):

    data_directory = './data/lunar/training/data/S12_GradeA/'
    mseed_file = f'{data_directory}{filename}.mseed'
    st = None
    try:
        st = obspy.read(mseed_file)
    except:
        print(f'Error reading {mseed_file}')
        continue

    print(st[0].stats)

    # This is how you get the data and the time, which is in seconds
    tr = st.traces[0].copy()
    tr_times = tr.times()
    tr_data = tr.data

    # # Start time of trace (another way to get the relative arrival time using datetime)
    # starttime = tr.stats.starttime.datetime
    # arrival = (arrival_time - starttime).total_seconds()
    # print(arrival)

    # # Initialize figure
    # fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    #
    # # Plot trace
    # ax.plot(tr_times, tr_data)
    #
    # # Mark detection
    # ax.axvline(x=arrival, color='red', label='Rel. Arrival')
    # ax.legend(loc='upper left')
    #
    # # Make the plot pretty
    # ax.set_xlim([min(tr_times), max(tr_times)])
    # ax.set_ylabel('Velocity (m/s)')
    # ax.set_xlabel('Time (s)')
    # ax.set_title(f'{mseed_file}', fontweight='bold')

    # Set the minimum frequency
    minfreq = 0.5
    maxfreq = 1.0

    # Going to create a separate trace for the filter data
    st_filt = st.copy()
    st_filt.filter('bandpass', freqmin=minfreq, freqmax=maxfreq)
    tr_filt = st_filt.traces[0].copy()
    tr_times_filt = tr_filt.times()
    tr_data_filt = tr_filt.data

    plt.figure(figsize=(10, 7))

    plt.plot(tr_times, tr_data, 'red', label='Original')
    plt.plot(tr_times_filt, tr_data_filt, 'blue', label='Filtered')
    # plt.axvline(x=arrival, color='black', label='Rel. Arrival')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.legend(loc='upper left')
    plt.show()
