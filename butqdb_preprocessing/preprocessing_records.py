import os
import argparse
import wfdb
from scipy.signal import butter, filtfilt

def downsample_signal(signal, initial_fs, target_fs):
    """downsample the ECG signal."""
    factor = int(initial_fs / target_fs)
    return signal[::factor]

def butter_bandpass_filter(signal, signal_fs=1000):
    """apply a Butterworth bandpass filter."""
    nyquist = 0.5 * signal_fs
    low = 0.5 / nyquist
    high = 40 / nyquist
    b, a = butter(2, [low, high], btype='band')
    return filtfilt(b, a, signal)

def process_record(record, data_path, filter_enabled, target_fs, overwrite):
    """process a single ECG record."""
    record_name = f"{record}_ECG"
    record_folder = os.path.join(data_path, record)
    record_path = os.path.join(record_folder, record_name)
    new_record_name = f"{record}_ECG_new"
    output_path = os.path.join(record_folder, new_record_name)

    # skip processing if overwrite is disabled and pre-processed file already exists
    if not overwrite and os.path.exists(f'{output_path}.dat'):
        print(f"Skipping {record}: Processed file already exists.")
        return

    try:
        # read the record
        r = wfdb.rdrecord(record_path, channels=[0])
        print(f"Processing record: {record}, Length: {r.sig_len}")

        signal = r.p_signal.flatten()

        # apply filter if requested
        if filter_enabled:
            signal = butter_bandpass_filter(signal, r.fs)

        # downsample if required
        if target_fs and r.fs != target_fs:
            signal = downsample_signal(signal, r.fs, target_fs)

        # save the processed signal
        wfdb.wrsamp(new_record_name, fs=target_fs or r.fs, sig_name=['ECG'], units=['mV'],
                    p_signal=signal.reshape([-1, 1]), write_dir=record_folder)

    except Exception as e:
        print(f"Error processing record {record}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Process ECG signals with filtering and downsampling options.")
    parser.add_argument('--filter', action='store_true', help='Enable Butterworth filter')
    parser.add_argument('--downsample', type=int, help='Target frequency for downsampling')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing processed files')
    args = parser.parse_args()

    BUTQDB_PATH = './data/but-qdb/brno-university-of-technology-ecg-quality-database-but-qdb-1.0.0'

    # get all record names from their folder names
    record_names = [d for _, dirs, _ in os.walk(BUTQDB_PATH) for d in dirs]

    for i, record in enumerate(record_names):
        print(f'({i+1}/{len(record_names)}) ', end='')
        process_record(record, BUTQDB_PATH, args.filter, args.downsample, args.overwrite)

if __name__ == "__main__":
    main()

# example usage:
# python preprocessing_records.py --filter --downsample 100 --overwrite