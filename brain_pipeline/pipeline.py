from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any, List, Dict
import multiprocessing as mp
import logging
from logging.handlers import QueueHandler, QueueListener
import re
import gc
import sys

from brain_pipeline.step.step import Step
from brain_pipeline.step.common import GC
from brain_pipeline import DefaultKeys, OptionalKey


def setup_worker_logger(log_queue: mp.Queue):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    queue_handler = QueueHandler(log_queue)
    logger.addHandler(queue_handler)
    return logger


def setup_main_logger(
    log_path: Path, log_queue: mp.Queue, overwrite=True
) -> tuple[Logger, QueueListener]:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if overwrite and log_path.exists():
        log_path.unlink()
    # formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(processName)s] [%(levelname)s] %(message)s"
    )
    # console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    # file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    # add queue listener
    listener = QueueListener(log_queue, console_handler, file_handler)
    queue_handler = QueueHandler(log_queue)
    logger.addHandler(queue_handler)
    listener.start()
    return logger, listener


class Pipeline:
    """
    Pipeline的工作原理
        一个Pipeline在初始化时需要指定多个流水线步骤，每一个步骤都是Step的子类；
    同时还需要指定输入和输出的键，这将用于存储从GlobDataloader中获取的数据。
        在运行时，Pipeline将会按照Steps中的步骤依次执行，每一个Step都需要指定
    输入和输出的键，用于从上下文中获取输入数据，并将输出数据存入上下文。
        在self.context，即上下文中，就保存了该流水线运行中当前时刻的全部数据。
    程序会读取在step中设定的输入和输出key，并对上下文进行管理。也就是说，在每一个
    step中，其输入都是上下文的子集，同样是键值对数据。而step的输出结果将会保存在
    上下文中，供后续使用。如果输出中有上下文中已经存在的键，则会覆盖原有的值。
        最后，Pipeline会返回一个字典，其中包含了输出键值对。
    """

    def __init__(
        self,
        steps: List[Step],
        input_keys: OptionalKey = None,
        output_keys: OptionalKey = None,
    ):
        self.steps = steps
        self.input_keys = (
            input_keys if input_keys is not None else [DefaultKeys.I_STI_PATH]
        )
        self.output_keys = (
            output_keys if output_keys is not None else [DefaultKeys.RETURN_CODE]
        )
        self.context: Dict[str, Any] = {}

    def __call__(self, init_data: Any, logger: Logger):
        self.context.clear()
        self.context[self.input_keys[0]] = init_data
        if DefaultKeys.RETURN_CODE not in self.output_keys:
            self.output_keys.append(
                DefaultKeys.RETURN_CODE
            )  # 确保OUTPUT_STATUS在输出key中
        for index, step in enumerate(self.steps):
            logger.info(
                f"Running step {step.__class__.__name__} ({index+1}/{len(self.steps)})"
            )
            input_data: Dict[str, Any] = {
                key: self.context.get(key, None) for key in step.input_keys
            }
            output_data = step(input_data, logger)
            if isinstance(output_data, GC):
                for key in output_data.keys():
                    self.context.pop(key, None)
                gc.collect()
            else:
                self.context.update(output_data)
        self.context[DefaultKeys.RETURN_CODE] = 0
        return {key: self.context.get(key, None) for key in self.output_keys}


class PipelineRunner:
    def __init__(
        self,
        pipeline: Pipeline,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        log_queue: mp.Queue,
        counter,
        lock,
    ):
        self.pipeline = pipeline
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.log_queue = log_queue
        self.counter = counter
        self.lock = lock

    def __call__(self):
        logger = setup_worker_logger(self.log_queue)
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
                self.output_queue.put({DefaultKeys.RETURN_CODE: 1})
            with self.lock:
                self.counter.value += 1
                logger.info(f"Processed {self.counter.value} files")


class GlobDataloader:
    def __init__(
        self,
        input_dir: Path,
        subfolder_pattern: str,
        filename_pattern: str,
        skip: int = 0,
    ):
        """
        根据指定的 subfolder_pattern 和 filename_pattern 来加载数据文件，是一个生成器
        :param input_dir: 输入文件目录
        :param subfolder_pattern: 子文件夹模式
        :param filename_pattern: 文件名模式
        :param skip: 跳过前 skip 个文件
        """
        self.input_dir = input_dir
        self.subfolder_pattern = subfolder_pattern
        self.filename_pattern = re.compile(filename_pattern)
        self.skip = skip
        logger = logging.getLogger()
        if self.skip > 0:
            logger.warning(f"Skipping first {self.skip} files")

    def __iter__(self):
        for filepath in self.input_dir.glob(self.subfolder_pattern):
            filename = filepath.name
            if self.filename_pattern.fullmatch(filename):
                if self.skip > 0:
                    self.skip -= 1
                    continue
                yield filepath


@dataclass
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    log_path: Path
    overwrite_log: bool


@dataclass
class ExecutionConfig:
    dataloader: GlobDataloader
    pipeline: Pipeline
    num_processes: int


def start(pipeline_config: PipelineConfig, executions: list["ExecutionConfig"]):
    """
    启动一个多进程的 pipeline 处理流程
    """
    if sys.platform == "win32":
        method = "spawn"
    else:
        method = "forkserver"
    mp.set_start_method(method, force=True)

    log_queue = mp.Queue()
    logger, listener = setup_main_logger(
        pipeline_config.log_path, log_queue, pipeline_config.overwrite_log
    )

    try:
        for idx, execution in enumerate(executions):
            logger.info(f"Start running pipeline: {idx+1} / {len(executions)}")
            # init variables
            input_queue: mp.Queue[Path | None] = mp.Queue()
            output_queue: mp.Queue = mp.Queue()
            counter = mp.Value("i", execution.dataloader.skip)
            lock = mp.Lock()
            # push files
            for filepath in execution.dataloader:
                input_queue.put(filepath)
            for _ in range(execution.num_processes * 2):
                input_queue.put(None)
            # start processes
            processes: list[mp.Process] = []
            for _ in range(execution.num_processes):
                runner = PipelineRunner(
                    execution.pipeline,
                    input_queue,
                    output_queue,
                    log_queue,
                    counter,
                    lock,
                )
                process = mp.Process(target=runner)
                process.start()
                logger.info(f"Started worker process {process.pid}")
                processes.append(process)
            # wait for processes
            for process in processes:
                process.join()
            # consume output queue
            output_queue.put(None)  # 结束标志
            occurred_exception: List[int] = []
            pointer = execution.dataloader.skip
            while True:
                output_item = output_queue.get()
                if output_item is None:
                    break
                pointer += 1
                if isinstance(output_item, dict):
                    return_code = output_item.get(DefaultKeys.RETURN_CODE, 1)
                else:
                    return_code = 1
                if return_code != 0:
                    occurred_exception.append(pointer)
            if len(occurred_exception) > 0:
                logger.warning(
                    f"When running the previous pipeline, exceptions occurred with the following files:\n"
                    f"{', '.join(map(str, occurred_exception))}"
                )
            else:
                logger.info(
                    f"When running the previous pipeline, no exception occurred."
                )
            # clean up
            input_queue.close()
            input_queue.join_thread()
            output_queue.close()
            output_queue.join_thread()
    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt received, terminating...")
    finally:
        listener.stop()
        log_queue.close()
        log_queue.join_thread()
