from pathlib import Path

from . import DATASET_RAW_DIR, DATASET_PROCESSED_DIR
from brain_pipeline.pipeline import (
    Pipeline,
    GlobDataloader,
    start,
    PipelineConfig,
    ExecutionConfig,
)
from brain_pipeline import DefaultKeys
from brain_pipeline.step.common.signal import ResamplePoly, SosFilter
from brain_pipeline.step.common.save import Save, FilepathFn
from brain_pipeline.step.common import GC
from brain_pipeline.step.stimuli.load import LoadStimuli
from brain_pipeline.step.stimuli.envelope import GammatoneEnvelope
from brain_pipeline.step.eeg.load import LoadEEG
from brain_pipeline.step.eeg.link import LinkEEG2Stimuli
from brain_pipeline.step.eeg.arrange import (
    InterpolateArtifacts,
    Align,
    RemoveArtifacts,
    CommonAverageReference,
)


ENVELOPE_64_BOARD_BAND = "envelope-64-board-band"
ENVELOPE_512_BOARD_BAND = "envelope-512-board-band"
EEG_64_BOARD_BAND = "eeg-64-board-band"
EEG_512_LOW_GAMMA = "eeg-512-low-gamma"


envelope_pipeline = Pipeline(
    steps=[
        LoadStimuli(),
        GammatoneEnvelope(),
        ResamplePoly(
            input_keys=[DefaultKeys.ENVELOPE_DATA, DefaultKeys.I_STI_SR],
            output_keys=[ENVELOPE_64_BOARD_BAND],
            target_sr=64,
        ),
        ResamplePoly(
            input_keys=[DefaultKeys.ENVELOPE_DATA, DefaultKeys.I_STI_SR],
            output_keys=[ENVELOPE_512_BOARD_BAND],
            target_sr=512,
        ),
        Save(
            input_keys=[DefaultKeys.I_STI_PATH, ENVELOPE_64_BOARD_BAND],
            filepath_fn=FilepathFn(DATASET_PROCESSED_DIR, "stimulus"),
        ),
        Save(
            input_keys=[DefaultKeys.I_STI_PATH, ENVELOPE_512_BOARD_BAND],
            filepath_fn=FilepathFn(DATASET_PROCESSED_DIR, "stimulus"),
        ),
    ],
    input_keys=[DefaultKeys.I_STI_PATH],
)

eeg_pipeline = Pipeline(
    steps=[
        LoadEEG(unit_multiplier=1e6, selected_channels=list(range(64))),
        LinkEEG2Stimuli(),
        SosFilter(None, None, 1, 0.5, "highpass", 1),
        InterpolateArtifacts(),
        Align(),
        RemoveArtifacts(),
        CommonAverageReference(),
        SosFilter(
            [DefaultKeys.TMP_EEG_DATA, DefaultKeys.I_EEG_SR],
            [EEG_64_BOARD_BAND],
            1,
            30,
            "lowpass",
            1,
        ),
        ResamplePoly(
            [EEG_64_BOARD_BAND, DefaultKeys.I_EEG_SR], [EEG_64_BOARD_BAND], 64, 1
        ),
        Save(
            [DefaultKeys.I_EEG_PATH, DefaultKeys.I_STI_PATH, EEG_64_BOARD_BAND],
            filepath_fn=FilepathFn(DATASET_PROCESSED_DIR, "eeg"),
        ),
        GC([EEG_64_BOARD_BAND]),
        SosFilter(
            [DefaultKeys.TMP_EEG_DATA, DefaultKeys.I_EEG_SR],
            [EEG_512_LOW_GAMMA],
            1,
            [35, 150],
            "bandpass",
            1,
        ),
        ResamplePoly(
            [EEG_512_LOW_GAMMA, DefaultKeys.I_EEG_SR], [EEG_512_LOW_GAMMA], 512, 1
        ),
        Save(
            [DefaultKeys.I_EEG_PATH, DefaultKeys.I_STI_PATH, EEG_512_LOW_GAMMA],
            filepath_fn=FilepathFn(DATASET_PROCESSED_DIR, "eeg"),
        ),
    ],
    input_keys=[DefaultKeys.I_EEG_PATH],
)


def main():
    config = PipelineConfig(
        input_dir=DATASET_RAW_DIR,
        output_dir=DATASET_PROCESSED_DIR,
        log_path=Path(__file__).parent / "preprocess.log",
        overwrite_log=True,
    )
    executions = [
        ExecutionConfig(
            dataloader=GlobDataloader(
                config.input_dir,
                "stimuli/eeg/*.npz.gz",
                r"^(podcast|audiobook).*\.npz\.gz$",
            ),
            pipeline=envelope_pipeline,
            num_processes=8,
        ),
        ExecutionConfig(
            dataloader=GlobDataloader(
                config.input_dir,
                "sub-*/*/eeg/*.bdf.gz",
                r"^sub-\d+_ses-[^_]+_task-(?!restingState)[^_]+_run-\d+_eeg\.bdf\.gz$",
            ),
            pipeline=eeg_pipeline,
            num_processes=4,
        ),
    ]
    start(config, executions)


if __name__ == "__main__":
    main()
