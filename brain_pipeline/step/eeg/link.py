from logging import Logger
from typing import Any, Dict
from pathlib import Path

import xml.etree.ElementTree as ElementTree

from brain_pipeline.step.step import Step
from brain_pipeline.step.stimuli.load import LoadStimuli, LoadStimuliTrigger
from brain_pipeline.default_keys import KeyTypeNoneOk, DefaultKeys


def load_apr(filepath: Path, logger: Logger) -> Dict[str, Any]:
    apr_data = {}
    tree = ElementTree.parse(filepath)
    root = tree.getroot()

    interactive_elements = root.findall(".//interactive/entry")
    for element in interactive_elements:
        description_element = element.find("description")
        assert description_element is not None
        if description_element.text == "SNR":
            snr = element.find("new_value")
            assert snr is not None
            apr_data[DefaultKeys.INPUT_APR_DATA_SNR] = snr.text
    if "snr" not in apr_data:
        logger.warning(f"Could not find SNR in {filepath}, defaulting to 100.")
        apr_data[DefaultKeys.INPUT_APR_DATA_SNR] = 100.0

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
    def __init__(self, input_keys: KeyTypeNoneOk = None, output_keys: KeyTypeNoneOk = None):
        super().__init__(
            input_keys,
            [DefaultKeys.INPUT_EEG_PATH],
            output_keys,
            [
                DefaultKeys.BASE_DIR,
                DefaultKeys.INPUT_STIMULI_PATH,
                DefaultKeys.INPUT_STIMULI_DATA,
                DefaultKeys.INPUT_STIMULI_SR,
                DefaultKeys.INPUT_STIMULI_TRIGGER_PATH,
                DefaultKeys.INPUT_STIMULI_TRIGGER_DATA,
                DefaultKeys.INPUT_STIMULI_TRIGGER_SR,
                DefaultKeys.INPUT_APR_PATH,
                DefaultKeys.INPUT_APR_DATA,
                DefaultKeys.INPUT_TSV_EVENTS_PATH,
                DefaultKeys.INPUT_TSV_EVENTS_DATA,
                DefaultKeys.INPUT_TSV_STIMULATION_PATH,
                DefaultKeys.INPUT_TSV_STIMULATION_DATA,
            ]
        )
        self.assert_keys('==', 1, '==', 13)

    def __call__(self, input_data: Dict[str, Any], logger: Logger) -> Dict[str, Any]:
        input_eeg_path: Path = input_data[self.input_keys[0]]
        base_dir: Path = input_eeg_path.parents[3]
        main_name = input_eeg_path.stem.split('_eeg')[0]
        apr_filepath = input_eeg_path.parent / f"{main_name}_eeg.apr"
        apr_data = load_apr(apr_filepath, logger)
        tsv_events_path = input_eeg_path.parent / f"{main_name}_events.tsv"
        tsv_events_data = load_tsv_events(tsv_events_path, logger)
        tsv_stimuli_path = input_eeg_path.parent / f"{main_name}_stimulation.tsv"
        tsv_stimuli_data = load_tsv_stimuli(tsv_stimuli_path, logger)

        stim_file = tsv_events_data["stim_file"]
        if stim_file == "n/a":
            raise ValueError(f"No stimulus file found in {tsv_events_path}")
        else:
            stimuli_path = base_dir / "stimuli" / stim_file
        stimuli_loader = LoadStimuli(input_keys=['tmp_in'], output_keys=['tmp_data', 'tmp_sr'])
        stimuli_dict = stimuli_loader({'tmp_in': stimuli_path}, logger)
        stimuli_data = stimuli_dict['tmp_data']
        stimuli_sr = stimuli_dict['tmp_sr']

        trigger_file = tsv_events_data["trigger_file"]
        if trigger_file == "n/a":
            raise ValueError(f"No trigger file found in {tsv_events_path}")
        else:
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
