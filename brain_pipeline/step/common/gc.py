from logging import Logger
from typing import Any, Dict

from brain_pipeline.step.step import Step
from brain_pipeline.default_keys import KeyTypeNotNone, KeyTypeNoneOk, DefaultKeys

class GC(Step):
    def __init__(self, input_keys: KeyTypeNotNone, output_keys: KeyTypeNoneOk = None):
        super().__init__(
            input_keys,
            [],
            output_keys,
            [DefaultKeys.GC_COLLECT]
        )
        self.assert_keys(">", 0, "==", 1)

    def __call__(self, input_data: Dict[str, Any], logger: Logger) -> Dict[str, Any]:
        return {
            DefaultKeys.GC_COLLECT: list(input_data.keys())
        }