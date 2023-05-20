from typing import Dict

import numpy as np

from rudalle.pipelines import generate_images
from rudalle import get_rudalle_model, get_tokenizer, get_vae

from src.engines.base import BaseEngine


class RuDalleEngine(BaseEngine):
    def __init__(
            self,
            config: Dict
    ):
        self.config = config

        self.model = get_rudalle_model(
            config['name'],
            pretrained=True,
            fp16=True,
            device=config['device'],
            cache_dir=config['path']
        )

        self.tokenizer = get_tokenizer()
        self.vae = get_vae(dwt=True).to(config['device'])

    def __call__(
            self,
            input: str,
            **kwargs
    ) -> Dict:
        pil_images, _ = generate_images(
            input,
            self.tokenizer,
            self.model,
            self.vae,
            top_k=self.config['top_k'],
            images_num=kwargs.get('images_num'),
            bs=self.config['bs'],
            top_p=self.config['top_p']
        )

        return {
            'images': [
                np.array(image)
                for image in pil_images
            ]
        }
