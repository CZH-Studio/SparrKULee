from abc import ABC, abstractmethod
from typing import Any, List, Dict
from logging import Logger
import operator

from brain_pipeline import Key, OptionalKey

class Step(ABC):
    def __init__(self, input_keys: OptionalKey, default_input_keys: Key,
                 output_keys: OptionalKey, default_output_keys: Key):
        self.input_keys: List[str] = input_keys if input_keys is not None else default_input_keys
        self.output_keys: List[str] = output_keys if output_keys is not None else default_output_keys
        self._operator = {
            '>': operator.gt,
            '<': operator.lt,
            '>=': operator.ge,
            '<=': operator.le,
            '==': operator.eq,
            '!=': operator.ne
        }
        # assert self.input_keys is not None and self.output_keys is not None

    @abstractmethod
    def __call__(self, input_data: Dict[str, Any], logger: Logger) -> Dict[str, Any]:
        raise NotImplementedError

    def assert_keys_num(self, op1: str, input_num: int, op2: str, output_num: int):
        assert self._operator[op1](len(self.input_keys), input_num), f"Input keys number {len(self.input_keys)} is not {op1} {input_num}"
        assert self._operator[op2](len(self.output_keys), output_num), f"Output keys number {len(self.output_keys)} is not {op2} {output_num}"
