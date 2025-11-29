from logging import Logger
from typing import Any, Dict
from pathlib import Path

import xml.etree.ElementTree as ElementTree

from brain_pipeline.step.step import Step
from brain_pipeline.step.stimuli.load import LoadStimuli, LoadStimuliTrigger
from brain_pipeline import OptionalKey, DefaultKeys


def load_apr(filepath: Path, logger: Logger) -> Dict[str, Any]:
    apr_data = {}
    tree = ElementTree.parse(filepath)
    root = tree.getroot()
    interactive = root.find('.//interactive')
    snr_value = None
    if interactive is not None:
        entries = interactive.findall('entry')
        for entry in entries:
            description_elem = entry.find('description')
            if description_elem is not None and description_elem.text == 'SNR':
                new_value_elem = entry.find('new_value')
                if new_value_elem is not None:
                    snr_text = new_value_elem.text
                    if snr_text is not None:
                        snr_value = float(snr_text)
                        break
    if snr_value is None:
        logger.warning(f"Could not find SNR in {filepath}, defaulting to 100.0")
        snr_value = 100.0
    else:
        logger.info(f"Found SNR value: {snr_value}")
    apr_data[DefaultKeys.I_APR_SNR] = snr_value
    return apr_data


def load_tsv_events(filepath: Path, logger: Logger):
    tsv_events_data = {}
    data_format_mapping = {
        "onset": float,
        "duration": float,
        "stim_file": str,
        "trigger_file": str,
        "noise_file": str,
        "video_file": str,
        "condition": str,
        "snr": float,
    }
    with open(filepath, 'r') as f:
        keys = f.readline().strip().split('\t')
        values = f.readline().strip().split('\t')
    for key, value in zip(keys, values):
        if key in data_format_mapping.keys():
            tsv_events_data[key] = data_format_mapping[key](value)
        else:
            logger.warning(f"Unknown key {key} in {filepath}")
    return tsv_events_data


def load_tsv_stimuli(filepath: Path, logger: Logger):
    tsv_stimuli_data = {}
    data_format_mapping = {
        "apx_file": str,
        "apr_file": str,
    }
    with open(filepath, 'r') as f:
        keys = f.readline().strip().split('\t')
        values = f.readline().strip().split('\t')
    for key, value in zip(keys, values):
        if key in data_format_mapping.keys():
            tsv_stimuli_data[key] = data_format_mapping[key](value)
        else:
            logger.warning(f"Unknown key {key} in {filepath}")
    return tsv_stimuli_data


class LinkEEG2Stimuli(Step):
    def __init__(self, input_keys: OptionalKey = None, output_keys: OptionalKey = None):
        """
        加载产生该EEG的刺激数据文件
        :param input_keys:
        :param output_keys:
        """
        super().__init__(
            input_keys,
            [DefaultKeys.I_EEG_PATH],
            output_keys,
            [
                DefaultKeys.BASE_DIR,
                DefaultKeys.I_STI_PATH,
                DefaultKeys.I_STI_DATA,
                DefaultKeys.I_STI_SR,
                DefaultKeys.I_STI_TRIGGER_PATH,
                DefaultKeys.I_STI_TRIGGER_DATA,
                DefaultKeys.I_STI_TRIGGER_SR,
                DefaultKeys.I_APR_PATH,
                DefaultKeys.I_APR_DATA,
                DefaultKeys.I_TSV_EVENTS_PATH,
                DefaultKeys.I_TSV_EVENTS_DATA,
                DefaultKeys.I_TSV_STI_PATH,
                DefaultKeys.I_TSV_STI_DATA,
            ]
        )
        self.assert_keys_num('==', 1, '==', 13)

    def __call__(self, input_data: Dict[str, Any], logger: Logger) -> Dict[str, Any]:
        # E:\Dataset\SparrKULee\sub-001\ses-shortstories01\eeg\sub-001_ses-shortstories01_task-listeningActive_run-01_eeg.bdf.gz
        input_eeg_path: Path = input_data[self.input_keys[0]]
        # E:\Dataset\SparrKULee
        base_dir: Path = input_eeg_path.parents[3]
        # sub-001_ses-shortstories01_task-listeningActive_run-01
        main_name = input_eeg_path.stem.split('_eeg')[0]
        # sub-001_ses-shortstories01_task-listeningActive_run-01_eeg.apr
        apr_filepath = input_eeg_path.parent / f"{main_name}_eeg.apr"
        apr_data = load_apr(apr_filepath, logger)
        # sub-001_ses-shortstories01_task-listeningActive_run-01_events.tsv
        tsv_events_path = input_eeg_path.parent / f"{main_name}_events.tsv"
        tsv_events_data = load_tsv_events(tsv_events_path, logger)
        # sub-001_ses-shortstories01_task-listeningActive_run-01_stimulation.tsv
        tsv_stimuli_path = input_eeg_path.parent / f"{main_name}_stimulation.tsv"
        tsv_stimuli_data = load_tsv_stimuli(tsv_stimuli_path, logger)

        stim_file = tsv_events_data["stim_file"]    # eeg/audiobook_5_1.npz.gz
        if stim_file == "n/a":
            raise ValueError(f"No stimulus file found in {tsv_events_path}")
        else:
            # E:\Dataset\SparrKULee\stimuli\eeg\audiobook_5_1.npz.gz
            stimuli_path = base_dir / "stimuli" / stim_file
        # 加载刺激数据
        stimuli_loader = LoadStimuli(input_keys=['tmp_in'], output_keys=['tmp_data', 'tmp_sr'])
        stimuli_dict = stimuli_loader({'tmp_in': stimuli_path}, logger)
        stimuli_data = stimuli_dict['tmp_data']
        stimuli_sr = stimuli_dict['tmp_sr']

        trigger_file = tsv_events_data["trigger_file"]  # eeg/t_audiobook_5_1.npz.gz
        if trigger_file == "n/a":
            raise ValueError(f"No trigger file found in {tsv_events_path}")
        else:
            # E:\Dataset\SparrKULee\stimuli\eeg\t_audiobook_5_1.npz.gz
            stimuli_trigger_path = base_dir / "stimuli" / tsv_events_data["trigger_file"]
        stimuli_trigger_loader = LoadStimuliTrigger(input_keys=['tmp_in'], output_keys=['tmp_data', 'tmp_sr'])
        stimuli_trigger_dict = stimuli_trigger_loader({'tmp_in': stimuli_trigger_path}, logger)
        stimuli_trigger_data = stimuli_trigger_dict['tmp_data']
        stimuli_trigger_sr = stimuli_trigger_dict['tmp_sr']

        ret = dict(zip(
            self.output_keys,
            [
                base_dir,
                stimuli_path, stimuli_data, stimuli_sr,
                stimuli_trigger_path, stimuli_trigger_data, stimuli_trigger_sr,
                apr_filepath, apr_data,
                tsv_events_path, tsv_events_data, tsv_stimuli_path, tsv_stimuli_data
            ])
        )
        return ret
