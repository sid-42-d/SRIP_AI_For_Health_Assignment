
import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

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
    for f in os.listdir(folder):
        if keyword.lower() in f.lower():
            return os.path.join(folder, f)
    raise FileNotFoundError(f"No file with keyword '{keyword}' in {folder}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', required=True)
    args = parser.parse_args()

    folder = args.name
    participant_id = os.path.basename(folder)
    out_dir = 'Visualizations'
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading signals for {participant_id}...")
    flow_df   = parse_signal_file(find_file(folder, 'Flow -'))
    thorac_df = parse_signal_file(find_file(folder, 'Thorac'))
    spo2_df   = parse_signal_file(find_file(folder, 'SPO2'))
    events    = parse_events_file(find_file(folder, 'Flow Events'))

    t0 = flow_df.index[0]
    def to_hours(df):
        return (df.index - t0).total_seconds() / 3600

    event_colors = {'Hypopnea': 'orange', 'Obstructive Apnea': 'red'}

    fig, axes = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
    fig.suptitle(f'Sleep Study - Participant {participant_id}', fontsize=14, fontweight='bold')

    signals = [
        (axes[0], to_hours(flow_df),   flow_df['value'],   'Nasal Airflow',     'steelblue'),
        (axes[1], to_hours(thorac_df), thorac_df['value'], 'Thoracic Movement', 'green'),
        (axes[2], to_hours(spo2_df),   spo2_df['value'],   'SpO2 (%)',          'purple'),
    ]

    for ax, t, vals, ylabel, color in signals:
        ax.plot(t, vals, color=color, linewidth=0.4, rasterized=True)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)
        for ev in events:
            ev_start_h = (ev['start'] - t0).total_seconds() / 3600
            ev_end_h   = (ev['end']   - t0).total_seconds() / 3600
            c = event_colors.get(ev['label'], 'gray')
            ax.axvspan(ev_start_h, ev_end_h, alpha=0.3, color=c, linewidth=0)

    axes[2].set_xlabel('Time (hours from start)', fontsize=9)
    legend_patches = [mpatches.Patch(color=c, alpha=0.6, label=l) for l, c in event_colors.items()]
    axes[0].legend(handles=legend_patches, loc='upper right', fontsize=8)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f'{participant_id}_visualization.pdf')
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()
