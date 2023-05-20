from typing import Dict

import numpy as np

from src.engines.torch_engine import TorchEngine


class Segmentator(TorchEngine):
    def __init__(
            self,
            config: Dict
    ):
        super().__init__(config=config, net_size=config['net_size'])

    def postprocessing(
            self,
            input: np.ndarray
    ) -> Dict:
        return {
            'mask': input.argmax(0)
        }
