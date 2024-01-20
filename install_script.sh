#!/bin/bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

conda create -n llm python=3.9 -y 
conda activate llm
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
conda install -c pytorch -c nvidia pytorch==2.0.1 pytorch-cuda=11.8 -y

PROTOC_ZIP=protoc-21.12-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
rm -f $PROTOC_ZIP

sudo apt-get install libssl-dev gcc -y

BUILD_EXTENSIONS=True make install

cd server
make install-awq
make install-eetq
make install-flash-attention
make install-flash-attention-v2-cuda
make install-vllm-cuda

cd flash-attention-v2/csrc
cd ft_attention && python setup.py install && cd ..
cd fused_dense_lib && python setup.py install && cd ..
cd fused_softmax && python setup.py install && cd ..
cd layer_norm && python setup.py install && cd ..
cd rotary && python setup.py install && cd ..
cd xentropy && python setup.py install && cd ..

cd ../../../

make run-ura-llama-7b
