from logging import Logger
from pathlib import Path
from typing import Any, List, Dict
import multiprocessing
import logging
from logging.handlers import QueueHandler, QueueListener
import re
import gc

from brain_pipeline.config import Config
from brain_pipeline.step.step import Step
from brain_pipeline.default_keys import DefaultKeys, KeyTypeNoneOk

def setup_worker_logger(log_queue: multiprocessing.Queue):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # 清除默认 handler

    queue_handler = QueueHandler(log_queue)
    logger.addHandler(queue_handler)


class Pipeline:
    def __init__(self, steps: List[Step], input_keys: KeyTypeNoneOk = None, output_keys: KeyTypeNoneOk = None):
        self.steps = steps
        self.input_keys = input_keys if input_keys is not None else [DefaultKeys.INPUT_STIMULI_PATH]
        self.output_keys = output_keys if output_keys is not None else [DefaultKeys.OUTPUT_STATUS]
        self.context: Dict[str, Any] = {}

    def __call__(self, input_data, logger: Logger):
        self.context.clear()
        self.context[self.input_keys[0]] = input_data
        for index, step in enumerate(self.steps):
            logger.info(f"Running step {step.__class__.__name__} ({index+1}/{len(self.steps)})")
            input_data: Dict[str, Any] = {key: self.context.get(key, None) for key in step.input_keys}
            output_data = step(input_data, logger)
            if DefaultKeys.GC_COLLECT in output_data.keys():
                for key in output_data[DefaultKeys.GC_COLLECT]:
                    try:
                        del self.context[key]
                    except KeyError:
                        logger.warning(f"Key {key} not found in context, skipping.")
                gc.collect()
            else:
                self.context.update(output_data)
        self.context[DefaultKeys.OUTPUT_STATUS] = 0
        return {key: self.context.get(key, None) for key in self.output_keys}


class PipelineRunner:
    def __init__(self, pipeline: Pipeline, input_queue, output_queue, log_queue, counter, lock):
        self.pipeline = pipeline
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.log_queue = log_queue
        self.counter = counter
        self.lock = lock

    def __call__(self):
        setup_worker_logger(self.log_queue)
        logger = logging.getLogger()

        while True:
            try:
                input_data = self.input_queue.get()
                if input_data is None:
                    break
                output = self.pipeline(input_data, logger)
                self.output_queue.put(output)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.exception(e)
            with self.lock:
                self.counter.value += 1
                logger.info(f"Processed {self.counter.value} files")


class GlobDataloader:
    def __init__(self, input_dir: Path, glob_pattern: str, re_pattern: str, skip: int = 0):
        self.input_dir = input_dir
        self.glob_pattern = glob_pattern
        self.re_pattern = re.compile(re_pattern)
        self.skip = skip
        if self.skip > 0:
            print(f"Skipping first {self.skip} files")

    def __iter__(self):
        for filepath in self.input_dir.glob(self.glob_pattern):
            filename = filepath.name
            if self.re_pattern.fullmatch(filename):
                if self.skip > 0:
                    self.skip -= 1
                    continue
                yield filepath


def setup_main_logger(log_path, log_queue: multiprocessing.Queue):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(processName)s] [%(levelname)s] %(message)s"))

    # 文件输出
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(processName)s] [%(levelname)s] %(message)s"))

    # 使用 QueueListener 来统一处理日志
    listener = QueueListener(log_queue, console_handler, file_handler)
    return listener


def start(config: Config, dataloaders: List[GlobDataloader], pipelines: List[Pipeline]):
    assert len(dataloaders) == len(pipelines), "Number of dataloaders and pipelines must be equal"

    multiprocessing.set_start_method('spawn')
    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()
    log_queue = multiprocessing.Queue()
    num_processes = config.num_processes if config.num_processes > 0 else multiprocessing.cpu_count()
    listener = setup_main_logger(config.log_path, log_queue)
    listener.start()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    queue_handler = QueueHandler(log_queue)
    logger.addHandler(queue_handler)

    for idx, (dataloader, pipeline) in enumerate(zip(dataloaders, pipelines)):
        logger.info(f'Start running pipeline: {idx+1} / {len(dataloaders)}')
        skip = dataloader.skip
        for filepath in dataloader:
            input_queue.put(filepath)
        for _ in range(num_processes):
            input_queue.put(None)
        counter = multiprocessing.Value('i', skip)
        lock = multiprocessing.Lock()
        processes = []
        for _ in range(num_processes):
            runner = PipelineRunner(pipeline, input_queue, output_queue, log_queue, counter, lock)
            process = multiprocessing.Process(target=runner)
            process.start()
            logger.info(f"Started worker process {process.pid}")
            processes.append(process)
        for process in processes:
            process.join()
