from base64 import b64decode, b64encode
import numpy as np
import cv2


def base64_to_image(b64_data: str) -> np.ndarray:
    im_arr = np.frombuffer(b64decode(b64_data.encode('utf-8')), dtype=np.uint8)
    return cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)


def image_to_base64(image: np.ndarray) -> str:
    _, im_arr = cv2.imencode('.jpg', image)
    return b64encode(im_arr.tobytes()).decode('utf-8')
