from logging import Logger
from typing import Any, Dict

from brain_pipeline.step.step import Step
from brain_pipeline import OptionalKey


class GC(Step):
    def __init__(self, input_keys: OptionalKey = None, output_keys: OptionalKey = None):
        """Trigger garbage collection

        :param input_keys: Values to be cleared, defaults to `[]`
        :type input_keys: OptionalKey
        :param output_keys: Empty, defaults to `[]`
        :type output_keys: OptionalKey
        """
        super().__init__(input_keys, [], output_keys, [])
        self.assert_keys_num(">=", 0, "==", 0)

    def __call__(self, input_data: Dict[str, Any], logger: Logger) -> Dict[str, Any]:
        return input_data
