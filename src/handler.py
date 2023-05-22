import json

import numpy as np
from navec import Navec
import cv2
from skimage.measure import label, regionprops

from src.engines import RuDalleEngine, Segmentator, Classificator
from src.utils import base64_to_image, image_to_base64


class Handler(object):
    def init(
            self,
            config_rudalle: str,
            config_classification: str,
            config_segmentation: str,
            config_common: str
    ):
        self.config_rudalle = json.load(open(config_rudalle, 'r'))
        self.config_classification = json.load(open(config_classification, 'r'))
        self.config_segmentation = json.load(open(config_segmentation, 'r'))
        self.config_common = json.load(open(config_common, 'r'))

        self.rudalle = RuDalleEngine(self.config_rudalle)
        self.classificator = Classificator(self.config_classification)
        self.segmentator = Segmentator(self.config_segmentation)
        self.navec = Navec.load(self.config_common['navec_path'])

        self._init_embeds()

    def _init_embeds(self):
        class_names = self.config_classification['classes']
        classes_ = [sum([_.split() for _ in class_name.split('-')], []) for class_name in class_names]
        classes_ = [[_.lower() for _ in class_name if len(_) > 2] for class_name in classes_]
        self.embeds = []
        for class_name in classes_:
            embed = np.zeros((300,), dtype=np.float32)
            for word in class_name:
                try:
                    embed_word = self.navec[word]
                except Exception:
                    embed_word = np.zeros((300,), dtype=np.float32)
                embed += embed_word
            self.embeds.append(embed / np.sqrt((embed * embed).sum()))

        self.embeds = np.stack(self.embeds)

    def __call__(
            self,
            image_str: str,
            query: str,
            target: str
    ) -> str:

        query_embed = np.zeros((300,), dtype=np.float32)
        for word in query.split():
            try:
                embed_word = self.navec[word]
            except Exception:
                embed_word = np.zeros((300,), dtype=np.float32)
            query_embed += embed_word
        query_embed /= np.sqrt((query_embed * query_embed).sum())
        class_idx = np.matmul(query_embed, self.embeds.T).argmax()
        qeury_class = self.config_classification['classes'][class_idx]

        image = base64_to_image(image_str)
        mask = self.segmentator(image)['mask'].astype('float32')
        import logging

        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        cv2.imwrite('/mnt/assets/models/test.jpg', mask.astype('uint8') * 255)
        label_mask = label(mask)
        props = regionprops(label_mask)

        bboxes = [prop.bbox for prop in props]
        emojies = [image[bbox[0]: bbox[2], bbox[1]: bbox[3], :] for bbox in bboxes]
        logging.error(bboxes)

        classes = [self.classificator(emoji)['class'] for emoji in emojies]
        logging.error(classes)
        logging.error(qeury_class)
        query_idxs = [idx for idx in range(len(classes)) if classes[idx] == qeury_class]
        logging.error(query_idxs)
        if len(query_idxs) == 0:
            return image_str

        emojies_gen = self.rudalle(target, images_num=len(query_idxs))['images']

        for i, idx in enumerate(query_idxs):
            bbox = bboxes[idx]
            emoji_gen = emojies_gen[i]
            emoji_gen = cv2.resize(emoji_gen, (bbox[3] - bbox[1], bbox[2] - bbox[0]))
            image[bbox[0]: bbox[2], bbox[1]: bbox[3], :] = emoji_gen

        return image_to_base64(image)
