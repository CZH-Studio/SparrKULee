from pathlib import Path

from brain_pipeline.pipeline import Pipeline
from brain_pipeline.default_keys import DefaultKeys
from brain_pipeline.step.common.signal import ResamplePoly, SosFilter
from brain_pipeline.step.common.save import Save, TargetFilePathFnFactory
from brain_pipeline.step.common.gc import GC
from brain_pipeline.step.stimuli.load import LoadStimuli
from brain_pipeline.step.stimuli.envelope import GammatoneEnvelope
from brain_pipeline.step.eeg.load import LoadEEG
from brain_pipeline.step.eeg.link import LinkEEG2Stimuli
from brain_pipeline.step.eeg.arrange import InterpolateArtifacts, Align, RemoveArtifacts, CommonAverageReference

ENVELOPE_64 = "envelope-64"
ENVELOPE_512 = "envelope-512"
EEG_64 = "eeg-64"
EEG_512 = "eeg-512"

envelope_pipeline = Pipeline(
    steps=[
        LoadStimuli(),

        GammatoneEnvelope(),
        ResamplePoly(
            input_keys=[DefaultKeys.ENVELOPE_DATA, DefaultKeys.INPUT_STIMULI_SR],
            output_keys=[ENVELOPE_64],
            target_sr=64
        ),
        SosFilter(
            [DefaultKeys.ENVELOPE_DATA, DefaultKeys.INPUT_STIMULI_SR],
            [ENVELOPE_512],
            1, [70, 220], "bandpass", 0
        ),
        ResamplePoly(
            input_keys=[ENVELOPE_512, DefaultKeys.INPUT_STIMULI_SR],
            output_keys=[ENVELOPE_512],
            target_sr=512
        ),
        Save(
            input_keys=[DefaultKeys.INPUT_STIMULI_PATH, ENVELOPE_64],
            target_file_path_fn=TargetFilePathFnFactory(
                Path(r"E:\Dataset\SparrKULee (preprocessed)"), "stimulus"
            )
        ),
        Save(
            input_keys=[DefaultKeys.INPUT_STIMULI_PATH, ENVELOPE_512],
            target_file_path_fn=TargetFilePathFnFactory(
                Path(r"E:\Dataset\SparrKULee (preprocessed)"), "stimulus"
            )
        )
    ],
    input_keys=[DefaultKeys.INPUT_STIMULI_PATH]
)

eeg_pipeline = Pipeline(
    steps=[
        LoadEEG(unit_multiplier=1e6, selected_channels=list(range(64))),
        LinkEEG2Stimuli(),
        SosFilter(
            None,
            None,
            1,0.5,"highpass",1
        ),
        InterpolateArtifacts(),
        Align(),
        RemoveArtifacts(),
        CommonAverageReference(),
        ResamplePoly(
            [DefaultKeys.TMP_EEG_DATA, DefaultKeys.INPUT_EEG_SR],
            [EEG_64],
            64,
            1
        ),
        Save(
            [DefaultKeys.INPUT_EEG_PATH, DefaultKeys.INPUT_STIMULI_PATH, EEG_64],
            target_file_path_fn=TargetFilePathFnFactory(
                Path(r"E:\Dataset\SparrKULee (preprocessed)"), "eeg"
            )
        ),
        GC([EEG_64]),
        SosFilter(
            [DefaultKeys.TMP_EEG_DATA, DefaultKeys.INPUT_EEG_SR],
            [DefaultKeys.TMP_EEG_DATA],
            1, [70, 220], "bandpass", 1
        ),
        ResamplePoly(
            [DefaultKeys.TMP_EEG_DATA, DefaultKeys.INPUT_EEG_SR],
            [EEG_512],
            512,
            1
        ),
        Save(
            [DefaultKeys.INPUT_EEG_PATH, DefaultKeys.INPUT_STIMULI_PATH, EEG_512],
            target_file_path_fn=TargetFilePathFnFactory(
                Path(r"E:\Dataset\SparrKULee (preprocessed)"), "eeg"
            )
        ),
    ],
    input_keys=[DefaultKeys.INPUT_EEG_PATH]
)