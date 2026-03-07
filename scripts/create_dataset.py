
import os
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

def parse_signal_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data_start = next(i for i, l in enumerate(lines) if l.strip() == 'Data:') + 1
    rows = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line:
            continue
        ts_part, val_part = line.split(';')
        ts_part = ts_part.strip().replace(',', '.')
        timestamp = pd.to_datetime(ts_part, format='%d.%m.%Y %H:%M:%S.%f')
        rows.append((timestamp, float(val_part.strip())))
    df = pd.DataFrame(rows, columns=['timestamp', 'value'])
    df.set_index('timestamp', inplace=True)
    return df

def parse_events_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    events = []
    for line in lines:
        line = line.strip()
        if not line or ';' not in line or '-' not in line.split(';')[0]:
            continue
        try:
            date_range, duration, label, *_ = line.split(';')
            date_part, time_range = date_range.strip().split(' ')
            t_start_str, t_end_str = time_range.split('-')
            start_dt = pd.to_datetime(date_part + ' ' + t_start_str.replace(',', '.'),
                                      format='%d.%m.%Y %H:%M:%S.%f')
            end_dt = pd.to_datetime(date_part + ' ' + t_end_str.replace(',', '.'),
                                    format='%d.%m.%Y %H:%M:%S.%f')
            if end_dt < start_dt:
                end_dt += pd.Timedelta(days=1)
            events.append({'start': start_dt, 'end': end_dt, 'label': label.strip()})
        except Exception:
            continue
    return events

def find_file(folder, keyword):
    keyword_lower = keyword.lower().strip()
    matches = []
    for f in os.listdir(folder):
        f_lower = f.lower().strip()
        if keyword_lower in f_lower:
            if keyword_lower == 'flow' and 'events' in f_lower:
                continue
            matches.append(f)
    if matches:
        return os.path.join(folder, sorted(matches, key=len)[0])
    raise FileNotFoundError(f"No file with keyword '{keyword}' in {folder}")

def bandpass_filter(signal, lowcut=0.17, highcut=0.4, fs=32, order=4):
    nyq  = fs / 2
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, signal)

def create_windows(signal, fs, window_sec=30, overlap=0.5):
    window_size = int(window_sec * fs)
    step_size   = int(window_size * (1 - overlap))
    windows, starts = [], []
    for start in range(0, len(signal) - window_size + 1, step_size):
        windows.append(signal[start:start + window_size])
        starts.append(start)
    return np.array(windows), starts

def label_windows(starts, fs, t0, events, window_sec=30):
    labels = []
    for start_idx in starts:
        win_start = t0 + pd.Timedelta(seconds=start_idx / fs)
        win_end   = win_start + pd.Timedelta(seconds=window_sec)
        label = 'Normal'
        for ev in events:
            overlap_sec = (min(win_end, ev['end']) - max(win_start, ev['start'])).total_seconds()
            if overlap_sec > 0.5 * window_sec:
                label = ev['label']
                break
        labels.append(label)
    return labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_dir',  required=True)
    parser.add_argument('-out_dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    participants = sorted([f for f in os.listdir(args.in_dir)
                           if os.path.isdir(os.path.join(args.in_dir, f))])
    all_data = []

    for pid in participants:
        folder = os.path.join(args.in_dir, pid)
        print(f"Processing {pid}...")

        flow_df   = parse_signal_file(find_file(folder, 'flow'))
        thorac_df = parse_signal_file(find_file(folder, 'thorac'))
        spo2_df   = parse_signal_file(find_file(folder, 'spo2'))
        events    = parse_events_file(find_file(folder, 'flow events'))

        flow_f   = bandpass_filter(flow_df['value'].values,   fs=32)
        thorac_f = bandpass_filter(thorac_df['value'].values, fs=32)
        spo2_f   = bandpass_filter(spo2_df['value'].values,   fs=4)

        flow_win,   flow_starts = create_windows(flow_f,   fs=32)
        thorac_win, _           = create_windows(thorac_f, fs=32)
        spo2_win,   _           = create_windows(spo2_f,   fs=4)

        t0     = flow_df.index[0]
        labels = label_windows(flow_starts, fs=32, t0=t0, events=events)

        for i in range(len(flow_win)):
            all_data.append({
                'participant': pid,
                'window_idx':  i,
                'flow':        flow_win[i],
                'thorac':      thorac_win[i],
                'spo2':        spo2_win[i],
                'label':       labels[i]
            })

        print(f"  {pid} done — {len(flow_win)} windows | {pd.Series(labels).value_counts().to_dict()}")

    full_df = pd.DataFrame(all_data)
    full_df['label'] = full_df['label'].replace('Mixed Apnea', 'Obstructive Apnea')
    full_df = full_df[full_df['label'] != 'Body event'].reset_index(drop=True)

    out_path = os.path.join(args.out_dir, 'full_dataset.pkl')
    full_df.to_pickle(out_path)
    print(f"\nTotal windows: {len(full_df)}")
    print(full_df['label'].value_counts())
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()
