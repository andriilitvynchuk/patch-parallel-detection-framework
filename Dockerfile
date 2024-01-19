FROM nvcr.io/nvidia/pytorch:20.11-py3
WORKDIR /app

RUN apt update && apt install git libgl1-mesa-glx libglib2.0-0 -y
COPY . .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -e . && \
    cd shared_numpy && python setup.py build_ext --inplace && pip install -e . && cd ..
