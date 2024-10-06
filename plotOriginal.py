# Import libraries
import numpy as np
import pandas as pd
from obspy import read
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
    csv_file = f'{data_directory}{filename}.csv'
    try:
        data_file = pd.read_csv(csv_file)
    except:
        print(f'Error reading {csv_file}')
        continue

    # Read in time steps and velocities
    csv_times = np.array(data_file['time_rel(sec)'].tolist())
    csv_data = np.array(data_file['velocity(m/s)'].tolist())

    # Plot the trace!
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(csv_times, csv_data)

    # Make the plot pretty
    ax.set_xlim([min(csv_times), max(csv_times)])
    ax.set_ylabel('Velocity (m/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'{filename}', fontweight='bold')

    # Plot where the arrival time is
    arrival_line = ax.axvline(x=arrival_time, c='red', label='Rel. Arrival')
    ax.legend(handles=[arrival_line])

    plt.show()
