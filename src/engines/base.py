from typing import Union, Dict

import numpy as np


class BaseEngine(object):
    def __call__(
            self,
            input: Union[np.ndarray, str],
            **kwargs
    ) -> Dict:
        raise NotImplementedError
