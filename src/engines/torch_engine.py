from typing import Dict

import numpy as np
import torch
import torchvision
from PIL import Image

from src.engines.base import BaseEngine


class TorchEngine(BaseEngine):
    def __init__(
            self,
            config: Dict,
            net_size: int
    ):
        self.config = config
        self.net_size = net_size

        self.model = torch.jit.load(config['path'], map_location=config['device']).eval()
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.net_size, self.net_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def postprocessing(
            self,
            input: np.ndarray
    ) -> Dict:
        raise NotImplementedError

    @torch.no_grad()
    def __call__(
            self,
            input: np.ndarray,
            **kwargs
    ) -> Dict:
        pil_image = Image.fromarray(input)
        tensor = self.transform(pil_image)
        output = self.model(tensor.unsqueeze(0).to(self.config['device'])).squeeze(0).cpu().numpy()
        return self.postprocessing(output)
