from logging import Logger
from typing import Any, Dict

from brain_pipeline.step.step import Step
from brain_pipeline import Key, OptionalKey

class GC(Step):
    def __init__(self, input_keys: Key, output_keys: OptionalKey = None):
        super().__init__(
            input_keys,
            [],
            output_keys,
            []
        )
        self.assert_keys_num(">", 0, "==", 0)

    def __call__(self, input_data: Dict[str, Any], logger: Logger) -> Dict[str, Any]:
        return input_data
