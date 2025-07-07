from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    num_processes: int
    input_dir: Path
    output_dir: Path
    log_path: Path
