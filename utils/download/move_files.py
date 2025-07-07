from pathlib import Path
from typing import Dict
import re
import shutil


def formatter(folder: str, *args) -> str:
    if len(args) == 0:
        return folder
    else:
        new_folder = str(folder).format(*args)
        return new_folder


def main():
    compiled_regex = {re.compile(pattern): folder for pattern, folder in regex_mapping.items()}
    for file in src_dir.glob('*'):
        moved = False
        for regex, folder in compiled_regex.items():
            if m := regex.fullmatch(file.name):
                folder = formatter(folder, *m.groups())
                dest_dir = tgt_dir / folder
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(file, dest_dir)
                moved = True
        if not moved:
            print('Could not find a match for {}'.format(file.name))

if __name__ == '__main__':
    src_dir = Path.home() / 'Downloads'
    tgt_dir = Path.home() / 'dataset' / 'SparrKULee'
    regex_mapping: Dict[str, str] = {
        r'^(\.bidsignore|\.bids-validator-config\.json|dataset_description\.json|participants\.(json|tsv)|README\.md|'
        r'task-audiogram_beh\.json|task-listeningActive_eeg\.json|task-listeningActive_events\.json|'
        r'task-listeningActive_stimulation\.json|task-restingState_events\.json)$': '',
        r'^sub-(\d+)_([0-9a-zA-Z-]+)_[0-9a-zA-Z-_]+\.npy$': 'derivatives/preprocessed_eeg/sub-{}/{}',
        r'^(audiobook|podcast)_[0-9a-zA-Z_]+\.(data_dict|npy)$': 'derivatives/preprocessed_stimuli',
        r'^(audiobook|noise_audiobook|noise_podcast|podcast|t_audiobook|t_podcast|)_[0-9a-zA-Z_]+\.(apx|npz\.gz|xml)$': 'stimuli/eeg',
        r'^sub-(\d+)_([0-9a-zA-Z-]+)_[0-9a-zA-Z-]+_[0-9a-zA-Z-]+_beh\.apr$': 'sub-{}/{}/beh',
        r'^sub-(\d+)_([0-9a-zA-Z-]+)_[0-9a-zA-Z-]+_beh\.tsv$': 'sub-{}/{}/beh',
        r'^sub-(\d+)_([0-9a-zA-Z-]+)_[0-9a-zA-Z-]+_[0-9a-zA-Z-]+_(eeg\.apr|eeg\.bdf\.gz|events\.tsv|stimulation\.tsv)$': 'sub-{}/{}/eeg',
        r'^sub-(\d+)_([0-9a-zA-Z-]+)_remarks\.(docx|txt)$': 'sub-{}/{}/remarks',
    }
    main()
