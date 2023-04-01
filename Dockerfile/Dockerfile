FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace
RUN apt-get update \
    && apt-get install -y tzdata build-essential git git-lfs python python3-pip libgl1-mesa-dev libglib2.0-0 \
    && unlink /usr/bin/python \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && unlink /usr/bin/pip \
    && ln -s /usr/bin/pip3 /usr/bin/pip

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 \
    && rm /tmp/requirements.txt

RUN git config --global --add safe.directory /workspace
