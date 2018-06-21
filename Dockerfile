FROM daviddocker78/mxnet-ssd:gpu_0.12.0_cuda9

ENV http_proxy http://132.222.121.98:8080/
ENV https_proxy http://132.222.121.98:8080/
ENV no_proxy=localhost,127.0.0.1

RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get install -y \
  wget \
  python-opencv \
  python-matplotlib \
  python-numpy \
  vim

RUN pip3 install opencv-python
RUN cd /mxnet/example/ssd/model && \
  wget https://github.com/zhreshold/mxnet-ssd/releases/download/v0.2-beta/vgg16_reduced.zip && \
  unzip vgg16_reduced.zip

RUN cd /mxnet/example/ssd/model && \
wget https://github.com/zhreshold/mxnet-ssd/releases/download/v0.6/resnet50_ssd_512_voc0712_trainval.zip && \
 unzip resnet50_ssd_512_voc0712_trainval.zip && \
 cd ../data/demo && \
 python download_demo_images.py


# install libcudnn 7.0.4.31
ENV CUDNN_VERSION 7.0.4.31
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
RUN apt-get update && apt-get install -y --allow-downgrades --no-install-recommends \
            libcudnn7=$CUDNN_VERSION-1+cuda9.0 \
            libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 \
  && rm -rf /var/lib/apt/lists/*

ENV BUILD_OPTS "USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1" 

#COPY akai/ /
#RUN cd /mxnet/example/ssd/data && \
#  ln -s /VOCdevkit .

