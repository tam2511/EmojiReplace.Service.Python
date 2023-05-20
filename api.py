from pathlib import Path
import os
from fastapi import FastAPI
from pydantic import BaseModel
import argparse
import uvicorn

from src.handler import Handler


def arg_parse():
    parser = argparse.ArgumentParser(
        description='Emoji api')
    parser.add_argument(
        'configs_path', type=Path,
        help='Path to configs')

    parser.add_argument(
        'port', type=int,
        help='Port')
    args = parser.parse_args()
    return args


app = FastAPI()

handler = Handler()
handler.init(
    os.path.join('configs', 'rudalle.json'),
    os.path.join('configs', 'classification.json'),
    os.path.join('configs', 'segmentation.json'),
    os.path.join('configs', 'common.json'),
)


class Input(BaseModel):
    image: str
    query: str
    target: str


class Output(BaseModel):
    image: str


@app.post("/predict", response_model=Output)
def embed(input: Input):
    return {
        'image': handler(input.image, input.query, input.target)
    }


if __name__ == '__main__':
    args = arg_parse()

    uvicorn.run("api:app", port=args.port, log_level="info", host='0.0.0.0')
