#!/bin/bash

# exp 1: forward vs backward
python trf_corr.py -e "eeg-256-board-band" -s "envelope-256-board-band"
python trf_corr.py -e "eeg-256-board-band" -s "envelope-256-board-band" -b

# exp 2: lag window
python trf_corr.py -e "eeg-256-board-band" -s "envelope-256-board-band" --lag_min_ms 0 --lag_max_ms 300
python trf_corr.py -e "eeg-256-board-band" -s "envelope-256-board-band" --lag_min_ms 0 --lag_max_ms 600

# exp 3: bandpass
python trf_corr.py -e "eeg-256-board-band" -s "envelope-256-board-band" --low 1 --high 8
python trf_corr.py -e "eeg-256-board-band" -s "envelope-256-board-band" --low 1 --high 12

# exp 4: shuffle control
python trf_corr.py -e "eeg-256-board-band" -s "envelope-256-board-band" --shift 5
