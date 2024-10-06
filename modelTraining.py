import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Bandpass filter design
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Function to pad signals to the maximum length in the dataset
def pad_signals(signals):
    return pad_sequences(signals, padding='post', dtype='float32')

# Sampling frequency in Hz
fs = 6.625
rangesFreq = [0.05, 0.1, 1]

# STA/LTA algorithm parameters
sta_len = 120  # Short-Term Average window length in seconds
lta_len = 600  # Long-Term Average window length in seconds
thr_on = 4
thr_off = 1.5

# Convert STA and LTA lengths to samples
sta_samples = int(np.round(sta_len * fs))
lta_samples = int(np.round(lta_len * fs))

# Define your databases
db = './data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv'
path = './data/lunar/training/data/'
df = pd.read_csv(db)
lsFilenames = df['filename'].tolist()

# Prepare lists to store signals and ROI labels
signals = []
roi_labels = []

for file in lsFilenames:
    print(f"Training model for database Luna")

    try:
        # Read the catalog file with the preserved column names
        csv_file = path + file + '.csv'
        data_cat = pd.read_csv(csv_file)
    except:
            print(f'Error reading {file}')
            continue
    # Extract time (relative) and velocity
    ls_time = np.array(data_cat['time_rel(sec)'].tolist())
    ls_velocity = np.array(data_cat['velocity(m/s)'].tolist())

    for i in range(len(rangesFreq) - 1):
        # Define the bandpass filter range
        fcl = rangesFreq[i] / fs  # Low cutoff frequency (normalized)
        fch = rangesFreq[i + 1] / fs  # High cutoff frequency (normalized)
        
        # Apply the bandpass filter
        bandpassFilterData = apply_bandpass_filter(ls_velocity, fcl, fch, fs, order=3)
        
        # Run STA/LTA trigger detection
        cft = classic_sta_lta(ls_velocity, sta_samples, lta_samples)
        on_off = np.array(trigger_onset(cft, thr_on, thr_off))
        
        # Store each signal and its ROI (start/end indices)
        for j in np.arange(0, len(on_off)):
            triggers = on_off[j]
            
            # Extract trigger start and end times
            start_index = triggers[0]
            end_index = triggers[1]
            
            # Store the signal and the corresponding ROI indices
            signals.append(ls_velocity)
            roi_labels.append([start_index, end_index])

    # Convert lists to arrays for further processing
    # Handle variable-length signals using padding
    padded_signals = pad_signals(signals)
    
    # Reshape padded_signals for Conv1D (add channel dimension)
    padded_signals = np.expand_dims(padded_signals, axis=-1)

    # Convert ROI labels to a NumPy array
    roi_labels = np.array(roi_labels)

    print(f"Training model on {len(padded_signals)} signals...")

    # Define the model architecture
    model = Sequential()
    
    # 1D Convolutional layer
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(padded_signals.shape[1], 1)))
    model.add(Flatten())
    
    # Dense layer
    model.add(Dense(100, activation='relu'))
    
    # Output layer for regression (predicting two values: start and end indices)
    model.add(Dense(2))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(padded_signals, roi_labels, epochs=10, batch_size=32, validation_split=0.2)

    # Save the model for this specific database
    model.save(f'model_{db[file]}.h5')

    print(f"Model for {db[file]} saved.")
