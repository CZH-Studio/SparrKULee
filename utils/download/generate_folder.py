from pathlib import Path

def get_ses(base: Path, i) -> Path:
    if 1 <= i <= 26:
        ses = base / 'ses-shortstories01'
    elif 27 <= i <= 31:
        ses = base / 'ses-varyingStories01'
    elif 32 <= i <= 36:
        ses = base / 'ses-varyingStories02'
    elif 37 <= i <= 42:
        ses = base / 'ses-varyingStories03'
    elif 43 <= i <= 46:
        ses = base / 'ses-varyingStories04'
    elif 47 <= i <= 48:
        ses = base / 'ses-varyingStories05'
    elif 49 <= i <= 56:
        ses = base / 'ses-varyingStories06'
    elif 57 <= i <= 62:
        ses = base / 'ses-varyingStories07'
    elif 63 <= i <= 71:
        ses = base / 'ses-varyingStories08'
    elif 72 <= i <= 78:
        ses = base / 'ses-varyingStories09'
    elif 79 <= i <= 85:
        ses = base / 'ses-varyingStories10'
    else:
        raise ValueError(f'Invalid value for i: {i}')
    return ses


def fn_derivatives():
    preprocessed_eeg = base_dir / 'derivatives' / 'preprocessed_eeg'
    for i in range(1, 86):
        sub = preprocessed_eeg / f'sub-{i:03d}'
        ses = get_ses(sub, i)
        ses.mkdir(parents=True, exist_ok=True)
    preprocessed_stimuli = base_dir / 'derivatives' / 'preprocessed_stimuli'
    preprocessed_stimuli.mkdir(parents=True, exist_ok=True)

def fn_stimuli():
    eeg = base_dir / 'stimuli' / 'eeg'
    eeg.mkdir(parents=True, exist_ok=True)

def fn_sub_xxx():
    for i in range(1, 86):
        sub = base_dir / f'sub-{i:03d}'
        ses = get_ses(sub, i)
        beh = ses / 'beh'
        eeg = ses / 'eeg'
        beh.mkdir(parents=True, exist_ok=True)
        eeg.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    base_dir = Path.home() / 'dataset' / 'SparrKULee'
    fn_derivatives()
    fn_stimuli()
    fn_sub_xxx()
