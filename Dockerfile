FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get install ffmpeg libsm6 git libxext6 htop rustc cargo  -y

RUN pip install Cython
RUN pip install --upgrade pip
RUN pip install rudalle

RUN pip install navec
RUN pip install fastapi
RUN pip install uvicorn
RUN pip install scikit-image

COPY configs configs
COPY src src
COPY api.py api.py

#RUN pip install notebook
#
#CMD jupyter notebook --ip='0.0.0.0' --NotebookApp.token='' --NotebookApp.password='' --port 11111 --allow-root --notebook-dir=/mnt