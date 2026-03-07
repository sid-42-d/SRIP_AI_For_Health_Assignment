# Sleep Breathing Irregularity Detection

## Overview
This project detects abnormal breathing patterns during sleep using physiological signals from 5 participants over 8-hour sleep sessions.

## Signals
- Nasal Airflow (32 Hz)
- Thoracic Movement (32 Hz)
- SpO2 / Oxygen Saturation (4 Hz)
- Flow Events — annotated breathing irregularities
- Sleep Profile — sleep stage annotations

## Project Structure
```
Project Root/
|-- Data/                  # Raw participant signal files
|-- Dataset/               # Processed windowed dataset
|-- Visualizations/        # Generated PDF plots      
|-- models/                # Model architecture
|-- notebooks/
|-- scripts/               # Processing and training scripts
|-- README.md
|-- requirements.txt
```

## How to Run
### Install dependencies
```
pip install -r requirements.txt
```
### Generate Visualization (one participant)
```
python scripts/vis.py -name "Data/AP01"
```
### Create Dataset (all participants)
```
python scripts/create_dataset.py -in_dir "Data" -out_dir "Dataset"
```
### Train and Evaluate
```
python scripts/train_model.py
```

## Methodology
- Bandpass filter applied to retain breathing frequency range (0.17-0.4 Hz)
- Signals split into 30-second windows with 50% overlap
- Each window labeled as Normal, Hypopnea, or Obstructive Apnea
- Simple 1D CNN trained and evaluated using Leave-One-Participant-Out Cross Validation

## Results
| Participant | Accuracy | Precision | Recall |
|-------------|----------|-----------|--------|
| AP01        | 0.4292   | 0.3668    | 0.6206 |
| AP02        | 0.7575   | 0.3850    | 0.6212 |
| AP03        | 0.1374   | 0.3308    | 0.3794 |
| AP04        | 0.7969   | 0.3544    | 0.3672 |
| AP05        | 0.4304   | 0.4002    | 0.4924 |
| **Average** | **0.5103** | **0.3674** | **0.4962** |
