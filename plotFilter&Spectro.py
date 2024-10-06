# Import libraries
import numpy as np
import pandas as pd
import obspy
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from scipy import signal
from matplotlib import cm

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

    # Set the minimum frequency
    minfreq = 0.5
    maxfreq = 1.0

    # Going to create a separate trace for the filter data
    st_filt = st.copy()
    st_filt.filter('bandpass', freqmin=minfreq, freqmax=maxfreq)
    tr_filt = st_filt.traces[0].copy()
    tr_times_filt = tr_filt.times()
    tr_data_filt = tr_filt.data

    f, t, sxx = signal.spectrogram(tr_data_filt, tr_filt.stats.sampling_rate)

    # Plot the time series and spectrogram
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(2, 1, 1)
    # Plot trace
    ax.plot(tr_times_filt, tr_data_filt)

    # Mark detection
    ax.axvline(x=arrival_time, color='red', label='Detection')
    ax.legend(loc='upper left')

    # Make the plot pretty
    ax.set_xlim([min(tr_times_filt),max(tr_times_filt)])
    ax.set_ylabel('Velocity (m/s)')
    ax.set_xlabel('Time (s)')

    ax2 = plt.subplot(2, 1, 2)
    vals = ax2.pcolormesh(t, f, sxx, cmap=cm.jet, vmax=5e-17)
    ax2.set_xlim([min(tr_times_filt), max(tr_times_filt)])
    ax2.set_xlabel(f'Time (Day Hour:Minute)', fontweight='bold')
    ax2.set_ylabel('Frequency (Hz)', fontweight='bold')
    ax2.axvline(x=arrival_time, c='red')
    cbar = plt.colorbar(vals, orientation='horizontal')
    cbar.set_label('Power ((m/s)^2/sqrt(Hz))', fontweight='bold')

    plt.show()
