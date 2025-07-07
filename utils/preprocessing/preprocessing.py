from pathlib import Path

from utils.preprocessing.pipeline import envelope_pipeline, eeg_pipeline
from brain_pipeline.config import Config
from brain_pipeline.pipeline import GlobDataloader, start


def main():
    config = Config(
        2,
        Path("E:/Dataset/SparrKULee"),
        Path("E:/Dataset/SparrKULee (preprocessed)"),
        Path(__file__).parent / "preprocessing.log"
    )
    dataloaders = [
        GlobDataloader(
            Path(config.input_dir / "stimuli"),
            "stimuli/eeg/*.npz.gz",
            r"^(podcast|audiobook).*\.npz\.gz$"
        ),
        GlobDataloader(
            Path(config.input_dir / "eeg"),
            "sub-*/*/eeg/*.bdf.gz",
            r'^sub-\d+_ses-[^_]+_task-(?!restingState)[^_]+_run-\d+_eeg\.bdf\.gz$',
            620
        )
    ]
    pipelines = [
        envelope_pipeline,
        eeg_pipeline
    ]
    start(config, dataloaders, pipelines)


if __name__ == '__main__':
    main()
