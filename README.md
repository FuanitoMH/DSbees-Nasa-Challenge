# DSbees-Nasa-Challenge
This is our solution for challenge 'Seismic Detection Across the Solar System' in the Nasa Space Apps Gto. 2024


# Seismic Signal Processing and ROI Detection using CNNs

This project processes seismic signals, applies STA/LTA trigger detection, and trains a 1D convolutional neural network (CNN) to detect regions of interest (ROI) in the signals.

## Overview

The code processes seismic signals from CSV files, applies a bandpass filter, and detects regions of interest (start and end points) using the Short-Term Average/Long-Term Average (STA/LTA) algorithm. A 1D CNN model is then trained on the filtered data to predict the start and end indices of signal triggers (ROI).

### Features

- Bandpass filtering for seismic data using Butterworth filters.
- STA/LTA algorithm to identify triggers in seismic signals.
- Padding of variable-length signals for uniform input to the CNN.
- A 1D Convolutional Neural Network (CNN) to detect the ROI in seismic signals.
- Support for saving trained models for each database.

## Requirements

- Python 3.7+
- NumPy
- Pandas
- SciPy
- ObsPy
- TensorFlow (Keras)

You can install the required packages by running:

```bash
pip install requirements.txt
```

## Code Structure

- **Bandpass Filtering**: The `apply_bandpass_filter` function uses a Butterworth filter to filter the seismic signal data within a specified frequency range.
- **STA/LTA Detection**: This algorithm detects seismic events by comparing short-term and long-term averages of the signal.
- **Padding Signals**: The `pad_signals` function pads variable-length signals to ensure uniform input shape for training.
- **1D CNN Model**: A 1D convolutional neural network is used to predict the start and end indices of detected triggers in the seismic signals.

## Usage

1. **Prepare Your Data**: Place your seismic signal data in CSV format under the `data/<database_name>/training/data/` directory. Each CSV should contain at least two columns: `time_rel(sec)` and `velocity(m/s)` for the seismic data.

2. **Run the Code**: Execute the script to:
    - Filter the seismic data with a bandpass filter.
    - Apply the STA/LTA algorithm to detect regions of interest (ROI).
    - Train a 1D CNN on the processed signals.
    - Save the trained model for future use.

3. **Training Details**:
    - The model is trained for 10 epochs with a batch size of 8, using 80% of the data for training and 20% for validation.
    - The model architecture consists of one 1D convolutional layer, followed by a flattening layer, and two dense layers for regression (predicting start and end points of triggers).

4. **Saving the Model**: The trained model is saved as `model_<database_name>.h5` in the current directory.

## Example Workflow

1. Organize your data in the following structure:

   ```
   └── data/
       └── <database_name>/
           └── training/
               └── data/
                   └── file1.csv
                   └── file2.csv
                   └── ...
   ```

2. Run the script:

   ```bash
   python modelTraining.py
   ```

3. The model will be trained and saved in the current directory for each database.

## License

This project is licensed under the MIT License.
